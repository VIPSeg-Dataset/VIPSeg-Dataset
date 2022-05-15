#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from detectron2.layers import Conv2d, get_norm
from .deform_conv_with_off import ModulatedDeformConvWithOff
import fvcore.nn.weight_init as weight_init



class ClipFuseHead(nn.Module):

    def __init__(self,cfg,in_dim, out_dim, clip_num, dilations,in_features):
        
        super().__init__()
        norm ="GN"
        use_bias = norm == ""
        self.in_dim = in_dim
        self.clip_num = clip_num
        self.convs = nn.ModuleList()
        self.convs.append(
            Conv2d(
                in_dim,
                out_dim,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, out_dim),
                activation=F.relu,
            )
        )
        
        weight_init.c2_xavier_fill(self.convs[-1])

        for dilation in dilations:
            self.convs.append(
                Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=use_bias,
                    norm=get_norm(norm, out_dim),
                    activation=F.relu,
                )
            )
            weight_init.c2_xavier_fill(self.convs[-1])

        image_pooling = nn.Sequential(
                nn.AvgPool2d(1),
                Conv2d(in_dim, out_dim, 1, bias=True, activation=F.relu),
            )
        weight_init.c2_xavier_fill(image_pooling[1])
        self.convs.append(image_pooling)

        self.cfg  =cfg
        if self.cfg.MODEL.TEMPORAL_KERNAL:
            self.register_parameter('t_w',nn.Parameter(torch.ones(clip_num)/clip_num))
        
        self.projects = nn.ModuleList()
        for ii in range(len(in_features)):
            self.projects.append( Conv2d(
                5 * out_dim+in_dim,
                in_dim,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, in_dim),
                activation=F.relu,
                ))
            weight_init.c2_xavier_fill(self.projects[-1])
        #self.convs_2

    def forward(self,features,in_features,frame_num):
        x = features['p3']
        size = x.shape[-2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=size, mode="bilinear", align_corners=False)
        res = torch.cat(res, dim=1)
        N,c,h,w = res.size()
        res = res.reshape(N//frame_num,frame_num,c,h,w)
    
        if self.cfg.MODEL.TEMPORAL_KERNAL:
            res = (res * self.t_w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(1,keepdim=True).repeat(1,frame_num,1,1,1)
        else:
            res = res.sum(1,keepdim=True).repeat(1,frame_num,1,1,1)/frame_num
        res = res.reshape(N,c,h,w)

        for ii,key in enumerate(in_features):
            feat = features[key]
            h,w = feat.shape[-2:]
            res_ = F.interpolate(res, size=(h,w), mode="bilinear", align_corners=False)
            feat = torch.cat([res_,feat],1)
            feat = self.projects[ii](feat)
            features[key] = feat
        return features


    



class SingleHead(nn.Module):
    """
    Build single head with convolutions and coord conv.
    """
    def __init__(self, in_channel, conv_dims, num_convs, deform=False, coord=False, norm='', name=''):
        super().__init__()
        self.coord = coord
        self.conv_norm_relus = []
        if deform:
            conv_module = ModulatedDeformConvWithOff
        else:
            conv_module = Conv2d
        for k in range(num_convs):
            conv = conv_module(
                    in_channel if k==0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
            self.add_module("{}_head_{}".format(name, k + 1), conv)
            self.conv_norm_relus.append(conv)

    def forward(self, x):
        if self.coord:
            x = self.coord_conv(x)
        for layer in self.conv_norm_relus:
            x = layer(x)
        return x
    
    def coord_conv(self, feat):
        with torch.no_grad():
            x_pos = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
            y_pos = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
            grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
            grid_x = grid_x.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
        feat = torch.cat([feat, grid_x, grid_y], dim=1)
        return feat


class PositionHead(nn.Module):
    """
    The head used in PanopticFCN for Object Centers and Stuff Regions localization.
    """
    def __init__(self, cfg):
        super().__init__()
        thing_classes   = cfg.MODEL.POSITION_HEAD.THING.NUM_CLASSES
        stuff_classes   = cfg.MODEL.POSITION_HEAD.STUFF.NUM_CLASSES
        bias_value      = cfg.MODEL.POSITION_HEAD.THING.BIAS_VALUE
        in_channel      = cfg.MODEL.FPN.OUT_CHANNELS
        conv_dims       = cfg.MODEL.POSITION_HEAD.CONVS_DIM
        num_convs       = cfg.MODEL.POSITION_HEAD.NUM_CONVS
        deform          = cfg.MODEL.POSITION_HEAD.DEFORM
        coord           = cfg.MODEL.POSITION_HEAD.COORD
        norm            = cfg.MODEL.POSITION_HEAD.NORM

        self.position_head = SingleHead(in_channel+2 if coord else in_channel, 
                                        conv_dims, 
                                        num_convs, 
                                        deform=deform,
                                        coord=coord,
                                        norm=norm,
                                        name='position_head')
        self.out_inst = Conv2d(conv_dims, thing_classes, kernel_size=3, padding=1)
        self.out_sem = Conv2d(conv_dims, stuff_classes, kernel_size=3, padding=1)
        for layer in [self.out_inst, self.out_sem]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, bias_value)

    def forward(self, feat):
        x = self.position_head(feat)
        x_inst = self.out_inst(x)
        x_sem = self.out_sem(x)
        return x_inst, x_sem


class KernelHead(nn.Module):
    """
    The head used in PanopticFCN to generate kernel weights for both Things and Stuff.
    """
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.FPN.OUT_CHANNELS
        conv_dims       = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        num_convs       = cfg.MODEL.KERNEL_HEAD.NUM_CONVS
        deform          = cfg.MODEL.KERNEL_HEAD.DEFORM
        coord           = cfg.MODEL.KERNEL_HEAD.COORD
        norm            = cfg.MODEL.KERNEL_HEAD.NORM

        self.kernel_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims,
                                      num_convs,
                                      deform=deform,
                                      coord=coord,
                                      norm=norm,
                                      name='kernel_head')
        self.out_conv = Conv2d(conv_dims, conv_dims, kernel_size=3, padding=1)
        nn.init.normal_(self.out_conv.weight, mean=0, std=0.01)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)
       
    def forward(self, feat):
        x = self.kernel_head(feat)
        x = self.out_conv(x)
        return x


class FeatureEncoder(nn.Module):
    """
    The head used in PanopticFCN for high-resolution feature generation.
    """
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.SEMANTIC_FPN.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        num_convs       = cfg.MODEL.FEATURE_ENCODER.NUM_CONVS
        deform          = cfg.MODEL.FEATURE_ENCODER.DEFORM
        coord           = cfg.MODEL.FEATURE_ENCODER.COORD
        norm            = cfg.MODEL.FEATURE_ENCODER.NORM
        
        self.encode_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims, 
                                      num_convs, 
                                      deform=deform,
                                      coord=coord,
                                      norm=norm, 
                                      name='encode_head')

    def forward(self, feat):
        feat = self.encode_head(feat)
        return feat


class ThingGenerator(nn.Module):
    """
    The head used in PanopticFCN for Things generation with Kernel Fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        input_channels  = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        self.sim_type   = cfg.MODEL.INFERENCE.SIMILAR_TYPE
        self.sim_thres  = cfg.MODEL.INFERENCE.SIMILAR_THRES
        self.class_spec = cfg.MODEL.INFERENCE.CLASS_SPECIFIC

        self.embed_extractor = Conv2d(input_channels, conv_dims, kernel_size=1)
        for layer in [self.embed_extractor]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, feat_shape, idx_feat, idx_shape, pred_cate=None, pred_score=None):
        #print(x.size())
        #print(feat_shape)
        #print(idx_feat.size())
        #print(idx_shape)
        #print(pred_cate)
        #print(pred_score.size())
#        exit()
         
        n, c, h, w = feat_shape
        if idx_shape>0:
         
            meta_weight = self.embed_extractor(idx_feat)
#            print(meta_weight.size())
            meta_weight = meta_weight.reshape(*meta_weight.shape[:2], -1)
            meta_weight = meta_weight.permute(0, 2, 1)
#            print(meta_weight.size())
            #exit()

            if not self.training:
                meta_weight, pred_cate, pred_score = self.kernel_fusion(meta_weight, pred_cate, pred_score)
            inst_pred = torch.matmul(meta_weight, x)
            inst_pred = inst_pred.reshape(n, -1, h, w)
#            print(inst_pred.size())
#            exit()
            return inst_pred, [pred_cate, pred_score]
        else:
            return [], [None, None]

    def kernel_fusion(self, meta_weight, pred_cate, pred_score):
        #print(meta_weight.size())
        #print(pred_cate.size())
        #print(pred_score.size())
        #print(pred_cate)
 
        meta_weight = meta_weight.squeeze(0)
        similarity = self.cal_similarity(meta_weight, meta_weight, sim_type=self.sim_type)
        #print(similarity.size())
        label_matrix = similarity.triu(diagonal=0) >= self.sim_thres
        #print(label_matrix.size())
        if self.class_spec:
            cate_matrix = pred_cate.unsqueeze(-1) == pred_cate.unsqueeze(0)
            label_matrix = label_matrix & cate_matrix
        cum_matrix = torch.cumsum(label_matrix.float(), dim=0) < 2
        keep_matrix = cum_matrix.diagonal(0)
        label_matrix = (label_matrix[keep_matrix] & cum_matrix[keep_matrix]).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(label_matrix, meta_weight) / label_norm
        pred_cate = pred_cate[keep_matrix]
        pred_score = pred_score[keep_matrix]
        #print(keep_matrix)
        #print(keep_matrix.sum())
        #exit()
        return meta_weight, pred_cate, pred_score

    def cal_similarity(self, base_w, anchor_w, sim_type="cosine"):
        if sim_type == "cosine":
            a_n, b_n = base_w.norm(dim=1).unsqueeze(-1), anchor_w.norm(dim=1).unsqueeze(-1)
            a_norm = base_w / a_n.clamp(min=1e-8)
            b_norm = anchor_w / b_n.clamp(min=1e-8)
            similarity = torch.mm(a_norm, b_norm.transpose(0, 1))
        elif sim_type == "L2":
            similarity = 1. - (base_w - anchor_w).abs().clamp(min=1e-6).norm(dim=1)
        else: raise NotImplementedError
        return similarity


class ConLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,meta_weight_for_con,thing_gt_idx,label_tmp):
        f_loss = 0.
        for b_idx,(meta_weight_b,thing_gt_b,label_tmp_b) in enumerate(zip(meta_weight_for_con,thing_gt_idx,label_tmp)):
            if (thing_gt_b.sum(1)>0).sum()<2:
                continue
            meta_weight_tmp = meta_weight_b[thing_gt_b.bool()]
            label_tmp_ = label_tmp_b[thing_gt_b.bool()]
            meta_weight_tmp = meta_weight_tmp.permute(1,0,2)
            for ii in range(len(meta_weight_tmp)):
                meta_weight_1 = meta_weight_tmp[ii]
                meta_weight_1 = F.normalize(meta_weight_1,dim=1)
                similarity_matrix = torch.matmul(meta_weight_1, meta_weight_1.T)
                label_matrix = label_tmp_.unsqueeze(0)==label_tmp_.unsqueeze(1)
                mask = torch.eye(label_matrix.shape[0], dtype=torch.bool).to(label_matrix.device)
                label_matrix_copy = label_matrix.clone()
                label_matrix[mask]=False
                positive_matrix = label_matrix
                negative_matrix = ~label_matrix_copy
                positives = []
                negatives = []
                ins_num_ = len(similarity_matrix)
   
                for jj in range(len(similarity_matrix)):
                    potential_pos = similarity_matrix[jj][positive_matrix[jj]]
                    if len(potential_pos)==0:
                        positives.append(similarity_matrix[jj][jj])
                    else:
                        idx_tmp = np.random.choice(len(potential_pos))
                        positives.append(potential_pos[idx_tmp])
                    
                    potential_neg = similarity_matrix[jj][negative_matrix[jj]]
                    while len(potential_neg)<(ins_num_-1):
                        potential_neg = torch.cat((potential_neg,potential_neg[-1].unsqueeze(0)),0)
                    negatives.append(potential_neg)
                positives = torch.stack(positives,0).reshape(ins_num_,-1)
                negatives = torch.stack(negatives,0)
                logits = torch.cat([positives, negatives], dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

                logits = logits / 0.07   
                f_loss+=self.criterion(logits,labels)        
 

        return f_loss

class VideoThingGenerator(nn.Module):
    """
    The head used in PanopticFCN for Things generation with Kernel Fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        input_channels  = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        self.sim_type   = cfg.MODEL.INFERENCE.SIMILAR_TYPE
        self.sim_thres  = cfg.MODEL.INFERENCE.SIMILAR_THRES
        self.class_spec = cfg.MODEL.INFERENCE.CLASS_SPECIFIC

        if cfg.TRAIN.USE_CON_LOSS:
            self.con_loss = ConLoss(cfg)
        self.embed_extractor = Conv2d(input_channels, conv_dims, kernel_size=1)
        for layer in [self.embed_extractor]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, feat_shape, idx_feat, idx_shape, thing_gt_idx=None,shuffle_times=None,pred_cate=None, pred_score=None,cfg=None):
#        print(x.size())
        #print(thing_gt_idx)
        #print(thing_gt_idx.size())
        #print(feat_shape)
        #print(idx_feat.size())
        #print(idx_shape)
        #print(pred_cate)
        #print(pred_score.size())
#        exit()
         
        if self.training:
            contrastive_loss =None
            inst_preds = []
            n, c, h, w = feat_shape
            if idx_shape>0:
             
                meta_weight = self.embed_extractor(idx_feat)
                if cfg.TRAIN.USE_CON_LOSS:
                    b_n,_,f_num = thing_gt_idx.size()
                    meta_weight_for_con = meta_weight.reshape(b_n,f_num,*meta_weight.size()[-3:]).permute(0,3,1,4,2)
                    label_tmp = torch.arange(_).unsqueeze(0).unsqueeze(-1).repeat(b_n,1,f_num).to(thing_gt_idx.device)
                    contrastive_loss = self.con_loss(meta_weight_for_con,thing_gt_idx,label_tmp)

                    
                if shuffle_times==1:
                    
                    meta_weight = meta_weight.reshape(*meta_weight.shape[:2], -1)
                    meta_weight = meta_weight.permute(0, 2, 1)
                    inst_pred = torch.matmul(meta_weight, x)
                    inst_pred = inst_pred.reshape(n, -1, h, w)
                    inst_preds.append(inst_pred)
                else:
                    #print(meta_weight.size())
                    for shuffle_t in range(shuffle_times):
                        if shuffle_t==0:
                            meta_weight_ = meta_weight.reshape(*meta_weight.shape[:2], -1)
                            meta_weight_ = meta_weight_.permute(0, 2, 1)
                            inst_pred = torch.matmul(meta_weight_, x)
                            inst_pred = inst_pred.reshape(n, -1, h, w)
                            inst_preds.append(inst_pred)
                        else:
                            b_n,_,f_num = thing_gt_idx.size()
            
                            index_t = torch.arange(f_num).unsqueeze(0).unsqueeze(0).expand_as(thing_gt_idx).to(thing_gt_idx.device)
                            for ii in range(b_n):
                                for jj in range(_):
                                    bool_ = thing_gt_idx[ii][jj]
                                    if bool_.sum()<=1:
                                        continue
                                    arr_idx = torch.arange(f_num).to(thing_gt_idx.device) 
                                    arr_idx_copy = arr_idx.clone()
                                    sel_idx = torch.nonzero(bool_).squeeze(1).long().cpu().numpy()
                                    sel_idx_shuffle = sel_idx.copy()
                                    np.random.shuffle(sel_idx_shuffle)
                                    arr_idx[sel_idx] = arr_idx_copy[sel_idx_shuffle]
                                    index_t[ii][jj] = arr_idx
            
                            meta_weight_ = meta_weight.reshape(b_n,f_num,*meta_weight.size()[-3:])
                            meta_weight_ = meta_weight_.permute(0,3,1,2,4)
                            meta_weight_clone =meta_weight_.clone()
     
                            for ii in range(b_n):
                                for jj in range(_):
                                    meta_weight_[ii][jj] =torch.index_select(meta_weight_clone[ii][jj],dim=0,index= index_t[ii][jj])
                            meta_weight_ = meta_weight_.permute(0,2,3,1,4)
                            meta_weight_ = meta_weight_.reshape(b_n*f_num,*meta_weight_.size()[-3:])
                            meta_weight_ = meta_weight_.reshape(*meta_weight.shape[:2], -1)
                            meta_weight_ = meta_weight_.permute(0, 2, 1)
                            inst_pred = torch.matmul(meta_weight_, x)
                            inst_pred = inst_pred.reshape(n, -1, h, w)
                            inst_preds.append(inst_pred)
                            
                return inst_preds, [pred_cate, pred_score],contrastive_loss
    
            else:
                for st in range(shuffle_times):
                    inst_preds.append([])
                return inst_preds, [None, None],None
        else:
            n, c, h, w = feat_shape
            if idx_shape>0:
    
                meta_weight = self.embed_extractor(idx_feat)
                meta_weight = meta_weight.reshape(*meta_weight.shape[:2], -1)
                meta_weight = meta_weight.permute(0, 2, 1)
    
                meta_weight, pred_cate, pred_score = self.kernel_fusion(meta_weight, pred_cate, pred_score)
                inst_pred = torch.matmul(meta_weight, x)
                inst_pred = inst_pred.reshape(n, -1, h, w)
    #            print(inst_pred.size())
    #            exit()
                return inst_pred, [pred_cate, pred_score,meta_weight]
            else:
                return [], [None, None,None]

    def kernel_fusion(self, meta_weight, pred_cate, pred_score):
        #print(meta_weight.size())
        #print(pred_cate.size())
        #print(pred_score.size())
        #print(pred_cate)
 
        meta_weight = meta_weight.squeeze(0)
        similarity = self.cal_similarity(meta_weight, meta_weight, sim_type=self.sim_type)
        #print(similarity.size())
        label_matrix = similarity.triu(diagonal=0) >= self.sim_thres
        #print(label_matrix.size())
        if self.class_spec:
            cate_matrix = pred_cate.unsqueeze(-1) == pred_cate.unsqueeze(0)
            label_matrix = label_matrix & cate_matrix
        cum_matrix = torch.cumsum(label_matrix.float(), dim=0) < 2
        keep_matrix = cum_matrix.diagonal(0)
        label_matrix = (label_matrix[keep_matrix] & cum_matrix[keep_matrix]).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(label_matrix, meta_weight) / label_norm
        pred_cate = pred_cate[keep_matrix]
        pred_score = pred_score[keep_matrix]
        #print(keep_matrix)
        #print(keep_matrix.sum())
        #exit()
        return meta_weight, pred_cate, pred_score

    def cal_similarity(self, base_w, anchor_w, sim_type="cosine"):
        if sim_type == "cosine":
            a_n, b_n = base_w.norm(dim=1).unsqueeze(-1), anchor_w.norm(dim=1).unsqueeze(-1)
            a_norm = base_w / a_n.clamp(min=1e-8)
            b_norm = anchor_w / b_n.clamp(min=1e-8)
            similarity = torch.mm(a_norm, b_norm.transpose(0, 1))
        elif sim_type == "L2":
            similarity = 1. - (base_w - anchor_w).abs().clamp(min=1e-6).norm(dim=1)
        else: raise NotImplementedError
        return similarity

class StuffGenerator(nn.Module):
    """
    The head used in PanopticFCN for Stuff generation with Kernel Fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        input_channels  = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        self.conv_dims  = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        
        self.embed_extractor = Conv2d(input_channels, self.conv_dims, kernel_size=1)
        for layer in [self.embed_extractor]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, feat_shape, idx_feat, idx_mask, pred_cate=None, pred_score=None):
        n, c, h, w = feat_shape
        meta_weight = self.embed_extractor(idx_feat)
        meta_weight = meta_weight.reshape(n, -1, self.conv_dims)
        if not self.training:
            meta_weight, pred_cate, pred_score = self.kernel_fusion(meta_weight, pred_cate, pred_score)
        seg_pred = torch.matmul(meta_weight, x)
        seg_pred = seg_pred.reshape(n, -1, h, w)
        return seg_pred, [pred_cate, pred_score]

    def kernel_fusion(self, meta_weight, pred_cate, pred_score):
        unique_cate = torch.unique(pred_cate)
        meta_weight = meta_weight.squeeze(0)
        cate_matrix, uniq_matrix = pred_cate.unsqueeze(0), unique_cate.unsqueeze(1)
        label_matrix = (cate_matrix == uniq_matrix).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(label_matrix, meta_weight) / label_norm
        pred_score = torch.mm(label_matrix, pred_score.unsqueeze(-1)) / label_norm
        return meta_weight, unique_cate, pred_score.squeeze(-1)

class VideoStuffGenerator(nn.Module):
    """
    The head used in PanopticFCN for Stuff generation with Kernel Fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        input_channels  = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        self.conv_dims  = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        
        self.embed_extractor = Conv2d(input_channels, self.conv_dims, kernel_size=1)
        for layer in [self.embed_extractor]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, feat_shape, idx_feat, idx_mask, pred_cate=None, pred_score=None):
        n, c, h, w = feat_shape
        meta_weight = self.embed_extractor(idx_feat)
#        meta_weight = meta_weight.reshape(n, -1, self.conv_dims)
        if not self.training:
            meta_weight = meta_weight.reshape(1, -1, self.conv_dims)
            meta_weight, pred_cate, pred_score = self.kernel_fusion(meta_weight, pred_cate, pred_score)
        else:
            meta_weight = meta_weight.reshape(n, -1, self.conv_dims)
        seg_pred = torch.matmul(meta_weight, x)
        seg_pred = seg_pred.reshape(n, -1, h, w)
        return seg_pred, [pred_cate, pred_score]

    def kernel_fusion(self, meta_weight, pred_cate, pred_score):
        unique_cate = torch.unique(pred_cate)
        meta_weight = meta_weight.squeeze(0)
        cate_matrix, uniq_matrix = pred_cate.unsqueeze(0), unique_cate.unsqueeze(1)
        label_matrix = (cate_matrix == uniq_matrix).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(label_matrix, meta_weight) / label_norm
        pred_score = torch.mm(label_matrix, pred_score.unsqueeze(-1)) / label_norm
        return meta_weight, unique_cate, pred_score.squeeze(-1)

def build_position_head(cfg, input_shape=None):
    return PositionHead(cfg)

def build_kernel_head(cfg, input_shape=None):
    return KernelHead(cfg)

def build_feature_encoder(cfg, input_shape=None):
    return FeatureEncoder(cfg)

def build_thing_generator(cfg, input_shape=None):
    return ThingGenerator(cfg)

def build_stuff_generator(cfg, input_shape=None):
    return StuffGenerator(cfg)
def build_thing_generator_video(cfg, input_shape=None):
    return VideoThingGenerator(cfg)

def build_stuff_generator_video(cfg, input_shape=None):
    return VideoStuffGenerator(cfg)

