import torch
from torch import nn
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .gt_generate import GenerateGT
from .v_gt_generate import VideoGenerateGT
from .loss import sigmoid_focal_loss, weighted_dice_loss
from .head import build_position_head, build_kernel_head, build_feature_encoder, build_thing_generator_video, build_stuff_generator_video, ClipFuseHead
from .backbone_utils import build_semanticfpn, build_backbone
from .utils import topk_score, multi_apply
from .data.dataset_mappers.utils import Video_BitMasks
__all__ = ["VideoPanopticFCN"]

@META_ARCH_REGISTRY.register()
class VideoPanopticFCN(nn.Module):
    """
    Implement PanopticFCN the paper :paper:`Fully Convolutional Networks for Panoptic Segmentation`.
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.device                = torch.device(cfg.MODEL.DEVICE)
        # parameters
        self.cfg                   = cfg
        self.ignore_val            = cfg.MODEL.IGNORE_VALUE
        self.common_stride         = cfg.MODEL.SEMANTIC_FPN.COMMON_STRIDE

        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        self.center_top_num        = cfg.MODEL.POSITION_HEAD.THING.TOP_NUM
        self.weighted_num          = cfg.MODEL.POSITION_HEAD.THING.POS_NUM
        self.center_thres          = cfg.MODEL.POSITION_HEAD.THING.THRES
        self.sem_thres             = cfg.MODEL.POSITION_HEAD.STUFF.THRES
        self.sem_classes           = cfg.MODEL.POSITION_HEAD.STUFF.NUM_CLASSES
        self.sem_with_thing        = cfg.MODEL.POSITION_HEAD.STUFF.WITH_THING
        self.in_feature            = cfg.MODEL.FEATURE_ENCODER.IN_FEATURES
        self.inst_scale            = cfg.MODEL.KERNEL_HEAD.INSTANCE_SCALES

        self.pos_weight            = cfg.MODEL.LOSS_WEIGHT.POSITION
        self.seg_weight            = cfg.MODEL.LOSS_WEIGHT.SEGMENT
        self.focal_loss_alpha      = cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma      = cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_GAMMA
        
        self.inst_thres            = cfg.MODEL.INFERENCE.INST_THRES
        self.panoptic_combine      = cfg.MODEL.INFERENCE.COMBINE.ENABLE
        self.panoptic_overlap_thrs = cfg.MODEL.INFERENCE.COMBINE.OVERLAP_THRESH
        self.panoptic_stuff_limit  = cfg.MODEL.INFERENCE.COMBINE.STUFF_AREA_LIMIT
        self.panoptic_inst_thrs    = cfg.MODEL.INFERENCE.COMBINE.INST_THRESH
        self.train_clipnum         = cfg.MODEL.TRAIN_CLIPNUM
        dilations                  = cfg.MODEL.DILATIONS
        self.shuffle_times         = cfg.INPUT.TRAIN_SHUFFLE_TIMES
        
        # backbone
        self.backbone              = build_backbone(cfg)
        self.semantic_fpn          = build_semanticfpn(cfg, self.backbone.output_shape())
        self.position_head         = build_position_head(cfg)
        self.kernel_head           = build_kernel_head(cfg)
        self.feature_encoder       = build_feature_encoder(cfg)
        self.thing_generator       = build_thing_generator_video(cfg)
        self.stuff_generator       = build_stuff_generator_video(cfg)
        self.get_ground_truth      = VideoGenerateGT(cfg)

        if cfg.MODEL.USE_CLIPFUSE:
            self.clip_fuse_head        = ClipFuseHead(cfg=cfg,in_dim = cfg.MODEL.KERNEL_HEAD.CONVS_DIM,out_dim =  cfg.MODEL.KERNEL_HEAD.CONVS_DIM//4,clip_num = self.train_clipnum,dilations = dilations,in_features = self.in_feature)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(1,3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(1,3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
            Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
        * "image": Tensor, image in (C, H, W) format.
        * "instances": Instances
        * "sem_seg": semantic segmentation ground truth.
        * Other information that's included in the original dicts, such as:
            "height", "width" (int): the output resolution of the model, used in inference.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:
                * "instances": Instances results.
                * "sem_seg": Semantic Segmentation results.
                * "panoptic_seg": available when `MODEL.INFERENCE.COMBINE.ENABLE`.
                  See the return value of
                  :func:`combine_thing_and_stuff` for its format.
        """
        images = [x["video_images"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        N, T, _, H, W  = images.tensor.size()

        images_input = images.tensor.reshape(N*T,-1,H,W)

        features = self.backbone(images_input)

        #print(images.tensor.size())
        #print(features.keys())
       
        encode_feat = self.semantic_fpn(features)
        encode_feat = self.feature_encoder(encode_feat)
        #print(encode_feat.size())
        #for k,v in features.items():
        #    print(k)
        #    print(v.size())
        if self.cfg.MODEL.USE_CLIPFUSE:
            features = self.clip_fuse_head(features,self.in_feature,T)
        #for k,v in features.items():
        #    print(k)
        #    print(v.size())
         
        #exit()
    
        features_in = [features[_feat] for _feat in self.in_feature]
        pred_centers, pred_regions, pred_weights = multi_apply(self.forward_single_level, features_in)

        if self.training:
            gt_dict = self.get_ground_truth.generate(batched_inputs, images, pred_weights, encode_feat, T)
            return self.losses(pred_centers, pred_regions, pred_weights, encode_feat, gt_dict,T,self.shuffle_times)
        else:
            return self.inference(batched_inputs, images, pred_centers, pred_regions, pred_weights, encode_feat,T)

    def forward_single_level(self, feature):
        pred_center, pred_region = self.position_head(feature)
        pred_weight = self.kernel_head(feature)
        #print(pred_center.size())
        #print(pred_region.size())
        #print(pred_weight.size())
        #exit()

        return pred_center, pred_region, pred_weight

    def losses(self, pred_centers, pred_regions, pred_weights, encode_feat, gt_dict,frame_num,shuffle_times):
        """
        Calculate losses of prediction with generated gt dict.

        Args:
            pred_centers: prediction for object centers
            pred_regions: prediction for stuff regions
            pred_weights: generated kernel weights for things and stuff
            encode_feat: encoded high-resolution feature
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "center": gt gaussian scoremap for things,
                "inst": gt instance target for things,
                "index": gt index for things,
                "index_mask": gt index mask for things,
                "class": gt classes for things,
                "sem_scores": gt semantic score map for stuff,
                "sem_labels":gt semantic target for stuff,
                "sem_index": gt index for stuff,
                "sem_masks": gt index mask for stuff,
            }

        Returns:
            loss(dict): a dict contains all information of loss function
            loss = {
                "loss_pos_th": position loss for things,
                "loss_pos_st": position loss for stuff,
                "loss_seg_th": segmentation loss for things,
                "loss_seg_st": segmentation loss for stuff,
            }
        """
        feat_shape = encode_feat.shape
        encode_feat = encode_feat.reshape(*feat_shape[:2], -1)
        frame_nums = []
        for ii in range(len(pred_centers)):
            frame_nums.append(frame_num)
       
        loss_pos_ths, loss_pos_sts, idx_feat_th, weighted_values, idx_feat_st, thing_nums, stuff_nums = \
                        multi_apply(self.loss_single_level, pred_centers,
                                    pred_regions, pred_weights,
                                    gt_dict["center"], gt_dict["inst"], 
                                    gt_dict["index_mask"], gt_dict["class"], 
                                    gt_dict["sem_scores"], gt_dict["sem_masks"], 
                                    gt_dict["sem_index"],frame_nums)

        #print(thing_nums)
        #print(stuff_nums)
        #exit()
        #print(idx_feat_th[0].size())
        #print(idx_feat_th[1].size())
        #print(idx_feat_th[2].size())
        #print(idx_feat_th[3].size())
        #print(idx_feat_th[4].size())

        thing_num = sum(thing_nums)
        stuff_num = sum(stuff_nums)
        idx_feat_th = torch.cat(idx_feat_th, dim=2)
        weighted_values = torch.cat(weighted_values, dim=1)
        idx_feat_st = torch.cat(idx_feat_st, dim=1)
        idx_feat_st = idx_feat_st.reshape(-1, *idx_feat_st.shape[2:])

        #print(encode_feat.size())
        #print(feat_shape)
        #print(thing_num)
        #print(stuff_num)
        #exit()
        thing_gt_idx = [_gt[:,:thing_nums[_idx],:] for _idx, _gt in enumerate(gt_dict["index_mask"])]
        thing_gt_idx = torch.cat(thing_gt_idx, dim=1)
 

        thing_preds, _, contrastive_loss = self.thing_generator(encode_feat, feat_shape, idx_feat_th, thing_num,thing_gt_idx,shuffle_times,cfg=self.cfg)
        stuff_pred, _ = self.stuff_generator(encode_feat, feat_shape, idx_feat_st, stuff_num)
        
        #print(stuff_pred.size())
        #exit()
        # for thing
         

        b_n,_,f_num = thing_gt_idx.size()
        thing_gt_idx = thing_gt_idx.permute(0,2,1).reshape(b_n*f_num,-1) 
        thing_gt_idx = thing_gt_idx.reshape(-1).bool()
        thing_gt_num = int(thing_gt_idx.sum())
        #print(gt_dict["inst"][0].size())
        thing_gt = [_gt[:,:thing_nums[_idx],...] for _idx, _gt in enumerate(gt_dict["inst"])]
        #print(thing_gt[0].size())
        thing_gt = torch.cat(thing_gt, dim=1)
        #print(thing_gt.size())
        thing_gt = thing_gt.permute(0,2,1,3,4).reshape(b_n*f_num,-1,*thing_gt.size()[-2:])
        #print(thing_gt.size())
        #print(thing_pred.size())
        #exit()
        loss_things = []
        for thing_pred in thing_preds:
            loss_thing = weighted_dice_loss(thing_pred, thing_gt, 
                                        gt_num=thing_gt_num,
                                        index_mask=thing_gt_idx,
                                        instance_num=thing_num,
                                        weighted_val=weighted_values,
                                        weighted_num=self.weighted_num,
                                        mode="thing",
                                        reduction="sum")        
            loss_things.append(loss_thing)
        # for stuff

        loss_things = sum(loss_things)/len(loss_things)

        stuff_gt_idx = [_gt[:,:stuff_nums[_idx]] for _idx, _gt in enumerate(gt_dict["sem_index"])]
        stuff_gt_idx = torch.cat(stuff_gt_idx, dim=1)
        stuff_gt_idx = stuff_gt_idx.permute(0,2,1).reshape(b_n*f_num,-1) 
        stuff_gt_idx = stuff_gt_idx.reshape(-1).bool()
        stuff_gt_num = int(stuff_gt_idx.sum())
   
        stuff_gt = [_gt[:,:stuff_nums[_idx],...] for _idx, _gt in enumerate(gt_dict["sem_labels"])]
        stuff_gt = torch.cat(stuff_gt, dim=1)
        stuff_gt = stuff_gt.permute(0,2,1,3,4).reshape(b_n*f_num,-1,*stuff_gt.size()[-2:])
        loss_stuff = weighted_dice_loss(stuff_pred, stuff_gt, 
                                        gt_num=stuff_gt_num,
                                        index_mask=stuff_gt_idx,
                                        instance_num=stuff_num,
                                        weighted_val=1.0,
                                        weighted_num=1,
                                        mode="stuff",
                                        reduction="sum")

        loss = {}
        # position loss
        loss["loss_pos_th"] = self.pos_weight * sum(loss_pos_ths) / max(thing_gt_num, 1)
        loss["loss_pos_st"] = self.pos_weight * sum(loss_pos_sts) / max(feat_shape[0],1)
        # segmentation loss
        loss["loss_seg_th"] = self.seg_weight * loss_things / max(thing_gt_num, 1)
        loss["loss_seg_st"] = self.seg_weight * loss_stuff / max(stuff_gt_num, 1)

        if  isinstance(contrastive_loss,torch.Tensor):
            loss['contrastive_loss']  =contrastive_loss*0.1
        else:
            loss['contrastive_loss']  =torch.zeros(1).to(loss_thing.device)
        return loss

    def loss_single_level(self, pred_center, pred_region, pred_weights, \
                          gt_center, gt_inst, gt_index_mask, gt_class, \
                          gt_sem_scores, gt_sem_masks, gt_sem_index,frame_num):
        
#        print(pred_center.size())
#        print(pred_region.size())
#        print(pred_weights.size())
#
#        print(gt_center.size())
#        print(gt_inst.size())
#        print(gt_index_mask.size())
#        #print(gt_index_mask)
#        #print(gt_class)
#        print(gt_class.size())
#        print(gt_sem_scores.size())
#        print(gt_sem_masks.size())
#        print(gt_sem_index.size())

        
        b_n,_,f_num,h_,w_ = gt_center.size()
        assert f_num == frame_num
        gt_center = gt_center.permute(0,2,1,3,4).reshape(b_n*f_num,-1,h_,w_)

        b_n,_,f_num,h_,w_ = gt_sem_scores.size()
        gt_sem_scores = gt_sem_scores.permute(0,2,1,3,4).reshape(b_n*f_num,-1,h_,w_)
        # position loss for things
        loss_pos_th = sigmoid_focal_loss(pred_center, gt_center,
                                         mode="thing",
                                         alpha=self.focal_loss_alpha,
                                         gamma=self.focal_loss_gamma,
                                         reduction="sum")
        # position loss for stuff
        loss_pos_st = sigmoid_focal_loss(pred_region, gt_sem_scores,
                                         mode="stuff",
                                         alpha=self.focal_loss_alpha,
                                         gamma=self.focal_loss_gamma,
                                         reduction="sum")
#        print(loss_pos_th)
#        print(loss_pos_st)
#        exit()
        # generate guided center
        batch_num, _, feat_h, feat_w = pred_center.shape
#        print(gt_inst.size())
#        print(torch.unique(gt_inst))
#        exit()
        b_n,_,f_num,h_,w_ = gt_inst.size()

        assert batch_num ==b_n*f_num
        gt_inst = gt_inst.permute(0,2,1,3,4).reshape(b_n*f_num,-1,h_,w_)

        guided_inst = F.interpolate(gt_inst, size=(feat_h, feat_w), mode='bilinear', align_corners=False)

        guided_inst = guided_inst.reshape(b_n,f_num,-1,feat_h,feat_w).permute(0,2,1,3,4)
       
        guidence = torch.zeros_like(guided_inst)
        pred_select = []
        pred_center = pred_center.reshape(b_n,f_num,-1,feat_h,feat_w).permute(0,2,1,3,4)
        for _idx in range(b_n):
            sub_pred = pred_center[_idx]
            sub_class = gt_class[_idx].to(torch.int64)
     
            sub_select = torch.index_select(sub_pred, dim=0, index=sub_class)
            pred_select.append(sub_select.sigmoid())
            
     

 
        pred_select = torch.stack(pred_select, dim=0)

        #print(pred_select.size())
        #print(guided_inst.size())
        #exit()
        keep = (guided_inst > 0.1) & (guided_inst < 255)
        guidence[keep] = pred_select[keep]

        weighted_values, guided_index = torch.topk(guidence.reshape(*guided_inst.shape[:3], -1), 
                                                   k=self.weighted_num, dim=-1)

        thing_num = int(max(gt_index_mask.max(2)[0].sum(dim=1).max(), 1))
        guided_index = guided_index[:,:thing_num,:, :]
                                 
        guided_index = guided_index.permute(0,2,1,3).reshape(batch_num, -1)
        weighted_values = weighted_values[:,:thing_num, :,:]
        # pred instance

#        _,_,h_,w_ = pred_weights.size()
#        pred_weights = pred_weights.reshape(b_n,frame_num,-1,h_,w_).permute(0,2,1,3,4)
        weight_shape = pred_weights.shape

        #print(weighted_values)
        #print(guided_index)
        
        inst_w = pred_weights.reshape(*weight_shape[:2], -1)
        idx_inst = guided_index.unsqueeze(1).expand(*weight_shape[:2], -1)

        #print(idx_inst)

        idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)
        idx_feat_th = idx_feat_th.reshape(*weight_shape[:2], thing_num, self.weighted_num)

        #print(idx_feat_th.size())
        #print('*'*100)
        # generate guided sem

        #print(gt_sem_index.size())
        #exit()
        stuff_num = int(max(gt_sem_index.max(2)[0].sum(dim=1).max(), 1))
        gt_sem_masks = gt_sem_masks[:, :stuff_num]
        b_n,_,f_num,h_,w_ = gt_sem_masks.size()
        gt_sem_masks = gt_sem_masks.permute(0,2,1,3,4).reshape(b_n*f_num,-1,h_,w_)
        gt_sem_masks = gt_sem_masks.unsqueeze(2)
        idx_feat_st = gt_sem_masks * pred_weights.unsqueeze(1)
        idx_feat_st = idx_feat_st.reshape(-1, *weight_shape[-3:])
        idx_feat_st = F.adaptive_avg_pool2d(idx_feat_st, output_size=1)
        idx_feat_st = idx_feat_st.reshape(batch_num, -1, weight_shape[1], 1, 1)

        return loss_pos_th, loss_pos_st, idx_feat_th, weighted_values, idx_feat_st, thing_num, stuff_num

    @torch.no_grad()
    def inference_single_level(self, pred_center, pred_region, pred_weights, pool_size):
        # pred things
        pred_center = pred_center.sigmoid()
        center_pool = F.avg_pool2d(pred_center, kernel_size=pool_size, 
                                    stride=1, padding=(pool_size-1)//2)
        pred_center = (pred_center + center_pool) / 2.0
        fmap_max = F.max_pool2d(pred_center, 3, stride=1, padding=1)
        keep = (fmap_max == pred_center).float()
        pred_center *= keep

        #print(keep.size())
#        print(pred_center.size())
        weight_shape = pred_weights.shape
        center_shape = pred_center.shape
        top_num = min(center_shape[-2]*center_shape[-1], self.center_top_num//2)

        sub_score, sub_index, sub_class, ys, xs = \
                topk_score(pred_center, K=top_num, score_shape=center_shape)


        idx_feat_ths, class_ths, score_ths = [], [], []
        thing_nums = []
        for frame_idx, (sub_score_,sub_index_,sub_class_,pred_weights_) in enumerate(zip(sub_score,sub_index,sub_class,pred_weights)):
            sub_score_ = sub_score_.unsqueeze(0)
            sub_index_ = sub_index_.unsqueeze(0)
            sub_class_ = sub_class_.unsqueeze(0)
            pred_weights_ = pred_weights_.unsqueeze(0)
       
            keep = sub_score_ > self.center_thres
            score_th = sub_score_[keep]
            class_th = sub_class_[keep]
            index = sub_index_[keep]
        
            index = index.unsqueeze(0).to(device=self.device, dtype=torch.long)
            thing_num = keep.sum()

            thing_nums.append(thing_num)
            if thing_num > 0:
                inst_w = pred_weights_.reshape(*pred_weights_.shape[:2], -1)
                idx_inst = index.unsqueeze(1).expand(*pred_weights_.shape[:2], -1)
                idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)
                idx_feat_th = idx_feat_th.unsqueeze(-1)
                idx_feat_ths.append(idx_feat_th)
                class_ths.append(class_th)
                score_ths.append(score_th)
#            else:
#                idx_feat_th, class_th, score_th = [], [], []
        if len(idx_feat_ths)>0:
            idx_feat_ths = torch.cat(idx_feat_ths,2)
            class_ths = torch.cat(class_ths, 0)
            score_ths = torch.cat(score_ths,0)
        thing_nums = sum(thing_nums)
        #    print(idx_feat_ths.size())
        #    print(class_ths.size())
        #    print(score_ths.size())
        #exit()
 
        
        # pred stuff
        pred_region = pred_region.sigmoid()


        b_n,_,h_,w_ = pred_region.size()
        pred_region = pred_region.permute(1,0,2,3).reshape(_,b_n*h_,w_).unsqueeze(0)
        pred_cate = pred_region.argmax(dim=1)

        class_st, num_class_st = torch.unique(pred_cate, return_counts=True)
        pred_st_mask = F.one_hot(pred_cate, num_classes=self.sem_classes)
        pred_st_mask = pred_st_mask.permute(0, 3, 1, 2).contiguous()


        score_st = (pred_region * pred_st_mask).reshape(1, self.sem_classes, -1)


        score_st = (score_st.sum(dim=-1)[:, class_st] / num_class_st).squeeze(0)
        pred_st_mask = pred_st_mask[:, class_st]
        keep = score_st > self.sem_thres
        stuff_num = keep.sum()
        score_st = score_st[keep]
        class_st = class_st[keep]
        pred_st_mask = pred_st_mask[:, keep]

        #exit()
        pred_st_mask = pred_st_mask.unsqueeze(2)

        b_n,_,h_,w_ = pred_weights.size()
        pred_weights_stuff = pred_weights.permute(1,0,2,3).reshape(_,b_n*h_,w_).unsqueeze(0)
        idx_feat_st = pred_st_mask * pred_weights_stuff.unsqueeze(1)
        idx_feat_st = idx_feat_st.reshape(-1, *pred_weights_stuff.shape[-3:])
        
        idx_feat_st = F.adaptive_avg_pool2d(idx_feat_st, output_size=1)
        if not self.sem_with_thing:
            class_st += 1

 
        return idx_feat_ths, class_ths, score_ths, thing_nums, idx_feat_st, score_st, class_st, stuff_num
    
    @torch.no_grad()
    def inference(self, batch_inputs, images, pred_centers_, pred_regions_, pred_weights_, encode_feats,frame_num):
        """
        Panoptic FCN inference process.

        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`
            image: ImageList in detectron2.structures
            pred_centers: prediction for object centers
            pred_regions: prediction for stuff regions
            pred_weights: generated kernel weights for things and stuff
            encode_feat: encoded high-resolution feature
        
        Returns:
            processed_results(dict): a dict contains all predicted results
            processed_results={
                "sem_seg": prediction of stuff for semantic segmentation eval, 
                "instances": prediction of things for instance segmentation eval,
                "panoptic_seg": prediction of both for panoptic segmentation eval.
            }
        """

        thing_ids_to_continue_dic = batch_inputs[0]['thing_ids_to_continue_dic']
        stuff_ids_to_continue_dic = batch_inputs[0]['stuff_ids_to_continue_dic']
        thing_cont_to_ids_dic = {}
        for k,v in thing_ids_to_continue_dic.items():
            thing_cont_to_ids_dic[v] = k 

        stuff_cont_to_ids_dic = {}
        for k,v in stuff_ids_to_continue_dic.items():
            stuff_cont_to_ids_dic[v] = k 

        result_img = batch_inputs[0]
        processed_results = []
        if "instances" in result_img.keys():
            img_shape = result_img["instances"].image_size
        else:
            img_shape = result_img["video_images"].shape[-2:]
        ori_shape = (result_img["height"], result_img["width"])

            


    #        print(len(results))
     
        encode_feat = encode_feats
 
        feat_shape = encode_feat.shape
        encode_feat = encode_feat.reshape(*feat_shape[:2], -1)
        result_instance = None

        assert encode_feat.size(0) ==frame_num
#        print(pred_regions_[0].size())
#        exit()
        pred_regions =  pred_regions_
        pred_weights =  pred_weights_
        pred_centers =  pred_centers_
        pool_size = [3,3,3,5,5]
        idx_feat_th, class_ths, score_ths, thing_num, idx_feat_st, score_sts, class_sts, stuff_num = \
                    multi_apply(self.inference_single_level, pred_centers,\
                        pred_regions, pred_weights, pool_size)
    
#            print(class_ths)
#            print(thing_num)
            #exit()
        thing_num = sum(thing_num)
        if thing_num == 0:
            result_instance = Instances(ori_shape, pred_masks=[], pred_boxes=[], 
                                        pred_classes=[], scores=[])
        else:
            class_ths = [_class for _class in class_ths if len(_class)>0]
            score_ths = [_score for _score in score_ths if len(_score)>0]
            idx_feat_th = [_feat for _feat in idx_feat_th if len(_feat)>0]


            #print(class_ths)

            class_ths = torch.cat(class_ths, dim=0)
            score_ths = torch.cat(score_ths, dim=0)
            idx_feat_th = torch.cat(idx_feat_th, dim=2)


            #print(class_ths)
            #print(class_ths.size())
        
            #print(idx_feat_th.size())

            keep = torch.argsort(score_ths, descending=True)
            #print(keep)
            #print(keep.size())
            idx_feat_th = idx_feat_th[:,:,keep]
            score_ths = score_ths[keep]
            class_ths = class_ths[keep]
            #print(idx_feat_th.size())
            #print(score_ths.size())
            #print(class_ths.size())
            #exit()

        stuff_num = sum(stuff_num)
        if stuff_num == 0:
            class_sts, idx_feat_st, score_sts = [], [], []
        else:
            score_sts = [_score for _score in score_sts if len(_score)>0]
            class_sts = [_cate_sem for _cate_sem in class_sts if len(_cate_sem)>0]
            idx_feat_st = [_feat for _feat in idx_feat_st if len(_feat)>0]
            score_sts = torch.cat(score_sts, dim=0)
            class_sts = torch.cat(class_sts, dim=0)
            idx_feat_st = torch.cat(idx_feat_st, dim=0)

        pred_thing, [class_ths, score_ths,thing_id_embs] = self.thing_generator(encode_feat, feat_shape, idx_feat_th, thing_num, pred_cate=class_ths, pred_score=score_ths)
        pred_stuff, [class_sts, score_sts] = self.stuff_generator(encode_feat, feat_shape, idx_feat_st, stuff_num, class_sts, score_sts)
        pred_stuff = pred_stuff.sigmoid()
        
#        print(class_ths.size())
#        print(score_ths.size())
#        print(thing_id_embs.size())
        if result_instance is None:
            result_instance, pred_mask, class_ths, score_ths,thing_id_embs = self.process_inst(
                        class_ths, score_ths, pred_thing, img_shape, ori_shape,thing_id_embs)                
#            print('*'*100)
#            print(class_ths.size())
#            print(score_ths.size())
#            print(thing_id_embs.size())
#            exit()
        else:
            pred_mask, class_ths, score_ths,thing_id_embs = None, None, None,None
        if self.sem_with_thing:
            sem_classes = self.sem_classes
        else:
            sem_classes = self.sem_classes + 1

        pred_stuff = F.interpolate(pred_stuff, scale_factor=self.common_stride, mode="bilinear", 
                                   align_corners=False)[...,:img_shape[0],:img_shape[1]]
        pred_stuff = F.interpolate(pred_stuff, size=ori_shape, mode="bilinear", align_corners=False)
        pred_sem_seg = torch.zeros(pred_stuff.size(0),sem_classes, *pred_stuff.shape[-2:], device=self.device)

#        print(pred_mask.size())
#        print(class_ths.size())
#        print(score_ths.size())
#        print(pred_stuff.size())
#        exit()
#
#        print(class_sts.size())
#        exit()
        pred_sem_seg[:,class_sts] += pred_stuff



#        processed_results.append({"sem_seg": pred_sem_seg, "instances": result_instance})

        if self.panoptic_combine:
            _,_,result_panoptic,dic_cat_idemb = self.combine_thing_and_stuff(
                [pred_mask, class_ths, score_ths,thing_id_embs],
                pred_sem_seg.argmax(dim=1),
                self.panoptic_overlap_thrs,
                self.panoptic_stuff_limit,
                self.panoptic_inst_thrs,
                thing_cont_to_ids_dic,
                stuff_cont_to_ids_dic,
                )
            #print(result_panoptic.size())
            #exit()
#            processed_results.append(result_panoptic.unsqueeze(0))

#        processed_results = torch.stack(processed_results)
#        print(processed_results.size())
#        exit()
        return result_panoptic,dic_cat_idemb

    @torch.no_grad()
    def process_inst(self, classes, scores, pred_inst, img_shape, ori_shape,thing_id_embs):
        """
        Simple process generate prediction of Things.

        Args:
            classes: predicted classes of Things
            scores: predicted scores of Things
            pred_inst: predicted instances of Things
            img_shape: input image shape
            ori_shape: original image shape
        
        Returns:
            result_instance: preserved results for Things
            pred_mask: preserved binary masks for Things
            classes: preserved object classes
            scores: processed object scores
        """

#        print(pred_inst.size())
#        print(classes.size())
#        print(scores.size())
#        exit()
        f_num,ins_num,h_,w_ = pred_inst.size()
        pred_inst = pred_inst.permute(1,0,2,3).reshape(1,ins_num,f_num*h_,w_)      
  
        pred_inst = pred_inst.sigmoid()[0]
        pred_mask = pred_inst > self.inst_thres
        # object rescore.
        sum_masks = pred_mask.sum((1, 2)).float() + 1e-6
        seg_score = (pred_inst * pred_mask.float()).sum((1, 2)) / sum_masks
        scores *= seg_score

        keep = torch.argsort(scores, descending=True)
        pred_inst = pred_inst[keep]
        pred_mask = pred_mask[keep]
        scores = scores[keep]
        classes = classes[keep]
        sum_masks = sum_masks[keep]
        thing_id_embs = thing_id_embs[keep]
        
        # object score filter.
        keep = scores >= 0.05
        if keep.sum() == 0:
            result_instance = Instances(ori_shape, pred_masks=[], pred_boxes=[], 
                                        pred_classes=[], scores=[])
   
            pred_mask = pred_mask.reshape(-1,f_num,h_,w_).permute(1,0,2,3)     
            return result_instance, pred_mask, None, None,None
        pred_inst = pred_inst[keep]
        scores = scores[keep]
        classes = classes[keep]
        thing_id_embs = thing_id_embs[keep]

        # sort and keep top_k
        keep = torch.argsort(scores, descending=True)
        keep = keep[:self.center_top_num]
        pred_inst = pred_inst[keep]
        scores = scores[keep].reshape(-1)
        classes = classes[keep].reshape(-1).to(torch.int32)
        thing_id_embs = thing_id_embs[keep]
        pred_inst = pred_inst.reshape(-1,f_num,h_,w_).permute(1,0,2,3)
        pred_inst = F.interpolate(pred_inst, 
                                  scale_factor=self.common_stride, 
                                  mode="bilinear", 
                                  align_corners=False)[...,:img_shape[0],:img_shape[1]]
        pred_inst = F.interpolate(pred_inst, 
                                  size=ori_shape, 
                                  mode="bilinear", 
                                  align_corners=False)

        pred_mask = pred_inst > self.inst_thres
        pred_bitinst = Video_BitMasks(pred_mask.permute(1,0,2,3))
        result_instance = Instances(ori_shape,
                                    pred_masks=pred_bitinst,
                             #       pred_boxes=pred_bitinst.get_bounding_boxes(),
                                    pred_classes=classes,
                                    scores=scores)
        return result_instance, pred_mask, classes, scores,thing_id_embs

    @torch.no_grad()
    def combine_thing_and_stuff(
        self,
        thing_results,
        stuff_results,
        overlap_threshold,
        stuff_area_limit,
        inst_threshold,
        thing_cont_to_ids_dic,
        stuff_cont_to_ids_dic,
    ):
        """
        Implement a simple combining logic following
        "combine_semantic_and_instance_predictions.py" in panopticapi
        to produce panoptic segmentation outputs.

        Args:
            thing_results: prediction of Things
            stuff_results: prediction of Stuff
            overlap_threshold: overlap threshold for Things combination
            stuff_area_limit: stuff area threshold for Stuff combination
            inst_threshold: instances confidence threshold

        Returns:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
            segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                Each dict contains keys "id", "category_id", "isthing".
        """
        pred_thing, thing_cate, thing_score,thing_id_embs = thing_results


        panoptic_seg = torch.zeros_like(stuff_results, dtype=torch.int32)
        panoptic_seg_mask = torch.ones_like(stuff_results, dtype=torch.int32)*(-1)
        current_segment_id = 0
        segments_info = []

        dic_tmp = {}

        if thing_cate is not None:
            pred_thing = pred_thing.permute(1,0,2,3)
            keep = thing_score >= inst_threshold
            if keep.sum() > 0:
                pred_thing = pred_thing[keep]
                thing_cate = thing_cate[keep]
                thing_score = thing_score[keep]
                thing_id_embs = thing_id_embs[keep]
                # Add instances one-by-one, check for overlaps with existing ones
                for _idx, (_mask, _cate, _score,id_emb) in enumerate(zip(pred_thing, thing_cate, thing_score,thing_id_embs)):
                    count=0
                    for frame_idx in range(panoptic_seg.size(0)):
                        _mask_ = _mask[frame_idx] 
                        mask_area = _mask_.sum().item()
                        intersect = _mask_ & (panoptic_seg[frame_idx] > 0)
                        intersect_area = intersect.sum().item()
                        if mask_area==0 or intersect_area * 1.0 / mask_area > overlap_threshold:
                            count+=1 
                            continue
                        if intersect_area > 0:
                            _mask_ = _mask_ & (panoptic_seg[frame_idx] == 0)
                    if count == len(panoptic_seg):
                        continue

                    current_segment_id += 1
                    panoptic_seg[_mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": True,
                            "score": _score.item(),
                            "category_id": thing_cont_to_ids_dic[_cate.item()],
                            "instance_id": _idx,
                        })
                    cat_id_ =  thing_cont_to_ids_dic[_cate.item()]
                    if (cat_id_,True) not in dic_tmp:
                        dic_tmp[(cat_id_,True)] = []
                        dic_tmp[(cat_id_,True)].append((current_segment_id,id_emb))
                    else:
                        if (current_segment_id,id_emb) not in  dic_tmp[(cat_id_,True)]:
                            dic_tmp[(cat_id_,True)].append((current_segment_id,id_emb))

        stuff_labels = torch.unique(stuff_results)
        #print(stuff_labels)
        #exit()
        for stuff_label in stuff_labels:
            if stuff_label == 0:  # 0 is a special "thing" class
                continue
            mask = (stuff_results == stuff_label) & (panoptic_seg == 0)
            mask_area = mask.sum()
            if mask_area < stuff_area_limit:
                continue
            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": False,
                    "category_id": stuff_cont_to_ids_dic[stuff_label.item()],
                    "area": mask_area.item(),
                })
            cat_id_ =  stuff_cont_to_ids_dic[stuff_label.item()]
            if (cat_id_,False) not in dic_tmp:
                dic_tmp[(cat_id_,False)] = []
                dic_tmp[(cat_id_,False)].append(current_segment_id)
            else:
                if current_segment_id not in  dic_tmp[(cat_id_,False)]:
                    dic_tmp[(cat_id_,False)].append(current_segment_id)

     
        dic_cat_idemb = {}
        for tmp_, curr_seg_id_list in dic_tmp.items():
            cat_id_,isthing = tmp_
            if isthing:
                dic_cat_idemb[cat_id_] =[]
                for ii,(cur_seg_id,id_emb_) in enumerate(curr_seg_id_list):
                    new_id = cat_id_*self.meta.label_divisor+ii
                    panoptic_seg_mask[panoptic_seg==cur_seg_id] = new_id
                    dic_cat_idemb[cat_id_].append(F.normalize(id_emb_,p=2,dim=0))
            else:
                for cur_seg_id in curr_seg_id_list:
                    panoptic_seg_mask[panoptic_seg==cur_seg_id] = cat_id_
            
    

        
        return panoptic_seg, segments_info, panoptic_seg_mask,dic_cat_idemb

    
