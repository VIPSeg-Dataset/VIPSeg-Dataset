#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
from functools import partial
import os
import sys
import torch
import numpy as np
import json
import time
from PIL import Image
from panopticapi.utils import rgb2id
from panopticapi.utils import IdGenerator

def topk_score(scores, K=40, score_shape=None):
    """
    get top K point in score map
    """
    batch, channel, height, width = score_shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

    #print(scores.size())
#
    #print(topk_scores.size())
    #print(topk_inds.size())
    #exit()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    # get all topk in in a batch
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)


    #print(topk_score.size())
  
    # div by K because index is grouped by K(C x K shape)
    #print(index)
    topk_clses = index // K
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

#    print(topk_inds.size())
#    print(topk_ys.size())
#    print(topk_xs.size())
#    #exit()
#
#    print(topk_clses)
#    exit()
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def generate_rgb_and_json(imgnames,img_tensors,meta,categories,save_folder,video_id):
    annotations = []
#    thing_dic = {}
#    for elm in total_segm_info:
#        thing_dic[elm['id']] = (elm['isthing'],elm['category_id'])
    categories = {el['id']: el for el in categories}
    color_generator = IdGenerator(categories)
    VOID = -1
    inst2color = {}
#    if len(total_segm_info) == 0:
#        for imgname, img_tensor in zip(imgnames,img_tensors):
#            print('processing image :{}'.format(imgname))
#            pan_format = np.zeros((img_tensor.shape[0], img_tensor.shape[1], 3), dtype=np.uint8)
#            segm_info = [{"category_id": 0, "iscrowd": 1, "id": 0,  "area": 0}]
#            #### save image
#            if not os.path.exists(os.path.join(save_folder,'pan_pred')):
#                os.makedirs(os.path.join(save_folder,'pan_pred'))
#            image_ = Image.fromarray(pan_format)
#            image_.save(os.path.join(save_folder,'pan_pred','_'.join(imgname.split('_')[:-1])+'.png'))
#
#
#            annotations.append({"segments_info": segm_info,"file_name":imgname})
#        return annotations

    if meta.name=='cityscapes_vps_video_train' or meta.name =='cityscapes_vps_image_train':
        div_ = 1000
    elif meta.name =='panoVSPW_vps_video_train':
        div_ = 125
    for imgname, img_tensor in zip(imgnames,img_tensors):
        print('processing image :{} {}'.format(video_id,imgname))
        pan_format = np.zeros((img_tensor.shape[0], img_tensor.shape[1], 3), dtype=np.uint8)
        l = np.unique(img_tensor)
        segm_info = []
        for el in l:
            if el == VOID:
                continue
            mask = img_tensor == el

            if el < div_:
                if el in meta.thing_dataset_id_to_contiguous_id.values():
                    continue
                else:
                    #stuff color
                    sem = el
                    if el in inst2color:
                        color = inst2color[el]

                    else:
                        color = color_generator.get_color(sem)
                        inst2color[el] = color

            else:
                ###thing color
                sem = int(el)//int(meta.label_divisor)


                if  sem not in meta.thing_dataset_id_to_contiguous_id.values():
                    print(l)
                    print('error')
                    print(sem)
                    print(el)
                    exit()
                if el in inst2color:
                    color = inst2color[el]
                else:
                    color = color_generator.get_color(sem)
                    inst2color[el] = color


            pan_format[mask] = color
            index = np.where(mask)
            x = index[1].min()
            y = index[0].min()
            width = index[1].max() - x
            height = index[0].max() - y

            dt = {"category_id": int(sem), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": int(mask.sum())}
            segm_info.append(dt)
            #### save image
        if not os.path.exists(os.path.join(save_folder,'pan_pred')):
            os.makedirs(os.path.join(save_folder,'pan_pred'))
        image_ = Image.fromarray(pan_format)
        if meta.name =='panoVSPW_vps_video_train':
            if not os.path.exists(os.path.join(save_folder,'pan_pred',video_id)):
                os.makedirs(os.path.join(save_folder,'pan_pred',video_id))
            image_.save(os.path.join(save_folder,'pan_pred',video_id,imgname.split('.')[0]+'.png'))
        else:
            image_.save(os.path.join(save_folder,'pan_pred','_'.join(imgname.split('_')[:-1])+'.png'))

        annotations.append({"segments_info": segm_info,"file_name":imgname})

    return annotations
