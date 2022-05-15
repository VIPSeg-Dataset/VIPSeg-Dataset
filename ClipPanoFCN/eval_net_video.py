#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""
import time
import os
import torch
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
import lap

import torch.nn.functional as F
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from panopticfcn import add_panopticfcn_config
from detectron2.data import MetadataCatalog, build_detection_train_loader,DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from panopticfcn import (
#    PanopticDeeplabDatasetMapper,
    PanopticDatasetVideoMapper,
    PanopticDatasetImageMapper,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from panopticfcn import VideoClipTestDataset,VideoClipTestNooverlapDataset
from panopticfcn.utils import generate_rgb_and_json


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs




class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)



    @classmethod
    def build_train_loader(cls, cfg):

        if cfg.INPUT.DATASET_MAPPER_NAME == "image_mapper":
            mapper = PanopticDatasetImageMapper(cfg,True)

        else:
            pass
        return build_detection_train_loader(cfg, mapper=mapper)



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
#    add_panoptic_deeplab_config(cfg)
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    cfg.defrost()
    #cfg.MODEL.WEIGHTS= os.path.join( args.pretrain_weight,'R-52.pkl')
#    cfg.MODEL.WEIGHTS= os.path.join( args.pretrain_weight,'model_final_coco.pkl')
    cfg.MODEL.WEIGHTS= os.path.join( args.pretrain_weight)
    cfg.SOLVER.IMS_PER_BATCH = args.img_per_batch
    cfg.MODEL.TRAIN_CLIPNUM = args.train_clipnum
    cfg.INPUT.CROP.ENABLED = False
    cfg.freeze()
    return cfg



def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    model.eval()


    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    annotations = []
    val_dataset = DatasetCatalog.get('panoVSPW_vps_video_'+args.split)
    print(len(val_dataset))


    thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values())
    thing_ids_to_continue_dic = {}
    for ii, id_ in enumerate(sorted(thing_ids)):
        thing_ids_to_continue_dic[id_] = ii


    stuff_ids=list(meta.stuff_dataset_id_to_contiguous_id.values())
    stuff_ids_to_continue_dic = {}
    for ii, id_ in enumerate(sorted(stuff_ids)):
        stuff_ids_to_continue_dic[id_] = ii+1


    for ii, video_log in enumerate(val_dataset):
        video_id = video_log['video_id']
#        if video_id !='1265_y-nJktexuuM':
#            continue 
        video_dataloader = VideoClipTestNooverlapDataset(video_log['file_names'],cfg.MODEL.TRAIN_CLIPNUM)
        imgnames_v = []
        
        final_preds = []

        reid_mem_dic = {}
        for jj in range(len(video_dataloader)):
            input_ = video_dataloader.getdata(jj)


            if cfg.MODEL.USE_CLIPFUSE and cfg.MODEL.TEMPORAL_KERNAL:
                imgs = input_['video_images']
                imgnames = input_['image_names']
                while len(imgnames)<cfg.MODEL.TRAIN_CLIPNUM:
                    imgs = torch.cat([imgs,imgs[-1].unsqueeze(0)],0)
                    imgnames.append(imgnames[-1])
                input_['video_images'] = imgs
                input_['image_names'] = imgnames
                   

            if jj ==0:
                imgnames_v = input_['image_names']
            else:
                imgnames_v.extend(input_['image_names'])

#            assert input_['video_images'].size(0)==2
            input_.update({'thing_ids_to_continue_dic':thing_ids_to_continue_dic,'stuff_ids_to_continue_dic':stuff_ids_to_continue_dic})

            input_ = [input_]

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                pano_preds,dic_cat_idemb = model(input_)
                torch.cuda.synchronize()
                end = time.time()
                print('stage1 time: {}'.format(end-start))

                
                if len(dic_cat_idemb)==0:
                    final_preds.append(pano_preds)
                    continue

                else:
                    if len(reid_mem_dic)==0:
                        reid_mem_dic = dic_cat_idemb
                        final_preds.append(pano_preds)
                        continue
                    else:
                        new_pano_preds = pano_preds.clone()
                        for cls_id in dic_cat_idemb:
                            if cls_id not in reid_mem_dic:
                                reid_mem_dic[cls_id] = dic_cat_idemb[cls_id]
                            else:
                                mem_feat = reid_mem_dic[cls_id]
                                mem_feat = torch.stack(mem_feat,0)
                                cur_feat = dic_cat_idemb[cls_id]
                                cur_feat =  torch.stack(cur_feat,0)

                                cos_dist = torch.matmul(cur_feat,mem_feat.t())
                                cos_dist = 1.-(cos_dist+1.)/2.
                                cost, x, y = lap.lapjv(cos_dist.cpu().numpy(), extend_cost=True, cost_limit=cfg.TEST.COST_THRESHOLD)
                                matches, unmatched_a, unmatched_b = [], [], []
                                for ix, mx in enumerate(x):
                                    if mx >= 0:
                                        matches.append([ix, mx])
                                unmatched_a = np.where(x < 0)[0]
                                unmatched_b = np.where(y < 0)[0]
                                matches = np.asarray(matches)
                                for matched in matches:
                                    cur_idx, mem_idx  = matched
                                    point_id = cls_id*meta.label_divisor+cur_idx
                                    ins_id = mem_idx
                                    new_id = cls_id* meta.label_divisor + ins_id
                                    new_pano_preds[pano_preds==point_id] = new_id
                                    reid_mem_dic[cls_id][mem_idx] = reid_mem_dic[cls_id][mem_idx]*(cfg.TEST.MEM_WEIGHT) + dic_cat_idemb[cls_id][cur_idx]*(1-cfg.TEST.MEM_WEIGHT)
                                    reid_mem_dic[cls_id][mem_idx] = F.normalize( reid_mem_dic[cls_id][mem_idx],p=2,dim=0)
                                for unmatched_idx in unmatched_a:
                                    ins_id = len(reid_mem_dic[cls_id])
                                    reid_mem_dic[cls_id].append(dic_cat_idemb[cls_id][unmatched_idx])
                                    new_id = cls_id* meta.label_divisor + ins_id
                                    point_id = cls_id*meta.label_divisor+unmatched_idx
                                    new_pano_preds[pano_preds==point_id] = new_id
                                
                         
                        final_preds.append(new_pano_preds)
                torch.cuda.synchronize()
                end2 = time.time()
                print('stage2 time {}'.format(end2-end))
                        
                



        final_preds = torch.cat(final_preds,0)
  
        final_preds_ = []                                                                                                                                                                                  
        imgnames_v_  =[]                                                                                                                                                                                   
        for imgname,f_p in zip(imgnames_v,final_preds):                                                                                                                                                    
            if imgname not in imgnames_v_:                                                                                                                                                                 
                imgnames_v_.append(imgname)                                                                                                                                                                
                final_preds_.append(f_p)                                                                                                                                                                   
                                                                                                                                                                                                           
        final_preds_ = torch.stack(final_preds_,0)  


        categories = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).categories
        annos = generate_rgb_and_json(imgnames_v_,final_preds_.squeeze(1).cpu().numpy(),meta,categories,args.imgsaveroot,video_id)
        annotations.append({'annotations':annos,'video_id':video_id})
    with open(os.path.join(args.imgsaveroot,'pred.json'),'w') as f:
        json.dump({'annotations':annotations},f)

                    
                





if __name__ == "__main__":
#    args = default_argument_parser().parse_args()
    parser = default_argument_parser()
    parser.add_argument(
        "--pretrain_weight",type=str)
    parser.add_argument(
        "--train_clipnum",type=int)
    parser.add_argument(
        "--img_per_batch",type=int)
    parser.add_argument(
        "--imgsaveroot",type=str)
    parser.add_argument(
        "--split",type=str,default='val')
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
