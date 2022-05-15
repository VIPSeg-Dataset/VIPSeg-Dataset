# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.projects.point_rend import ColorAugSSDTransform
from panopticapi.utils import rgb2id

from panoptic_deeplab.target_generator import VideoPanopticDeepLabTargetGenerator

#from .utils import Video_BitMasks

__all__ = ["PanopticDatasetVideoMapper"]


class PanopticDatasetVideoMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        #size_divisibility,
        train_clipnum,
        panoptic_target_generator,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        #self.size_divisibility = size_divisibility
        self.train_clipnum  =train_clipnum
        self.panoptic_target_generator = panoptic_target_generator

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
#        if cfg.INPUT.COLOR_AUG_SSD:
#            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        panoptic_target_generator = VideoPanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )



        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            #"size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "train_clipnum": cfg.INPUT.TRAIN_CLIPNUM,
             "panoptic_target_generator":panoptic_target_generator,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        #print(dataset_dict['width'])
        #exit()

        #print(dataset_dict)
        video_length = len(dataset_dict['file_names'])
        assert self.train_clipnum < video_length
        choice_index = np.random.choice(video_length-self.train_clipnum,1)        
        choice_index = int(choice_index[0])
        
        select_filenames = []
        #select_sem_seg_file_names = [] 
        select_pan_seg_file_names = []
        select_segments_infos = []
         
         
        for idx in range(choice_index,choice_index+self.train_clipnum):
            select_filenames.append(dataset_dict['file_names'][idx])
#            if "sem_seg_file_names" in dataset_dict:
#                select_sem_seg_file_names.append(dataset_dict['sem_seg_file_names'][idx])
#            else:
#                select_sem_seg_file_names.append(None)
            if "pan_seg_file_names" in dataset_dict:
                select_pan_seg_file_names.append(dataset_dict["pan_seg_file_names"][idx])
                select_segments_infos.append(dataset_dict['segments_infos'][idx])
                
            else:
                select_pan_seg_file_names.append(None)
                select_segments_infos.append(None)
        ######################
        out_dic={}

        FLAG=0
        count = 0
#        while FLAG==0 and count<10:
            #print(count)
#            count+=1
        #    print(count)
        insid_catid_dic={}
        input_images=[]
#        input_semantic_seg=[]
        input_panoptic_seg=[]
        for ii_, (file_name,pan_seg_file_name,segments_infos) in enumerate(zip(select_filenames,select_pan_seg_file_names,select_segments_infos)):

            ######
            for segments_info in segments_infos:
                class_id = segments_info["category_id"]
                ins_id = segments_info['id']
                if not segments_info['iscrowd']:
                    if ins_id not in insid_catid_dic:
                        insid_catid_dic[ins_id] = class_id
            #####
            if ii_ ==0:
                image =  utils.read_image(file_name, format=self.img_format)
                utils.check_image_size(dataset_dict, image)
                if pan_seg_file_name is not None:
                    pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
                else:
                    pan_seg_gt = None
                    raise ValueError(
                        "Cannot find 'pan_seg_file_names' for panoptic segmentation dataset."
                        )

                aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                image = aug_input.image
                pan_seg_gt = aug_input.sem_seg

            else:
                image =  utils.read_image(file_name, format=self.img_format)
                utils.check_image_size(dataset_dict, image)
                image = transforms.apply_image(image)
                if pan_seg_file_name is not None:
                    pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
                else:
                    pan_seg_gt = None
                    raise ValueError(
                        "Cannot find 'pan_seg_file_names' for panoptic segmentation dataset."
                        )

                 
            
            # apply the same transformation to panoptic segmentation
                pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)


            pan_seg_gt = rgb2id(pan_seg_gt)

            # Pad image and segmentation label here!
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                
            input_images.append(image.unsqueeze(0))
            input_panoptic_seg.append(pan_seg_gt)
        

        input_images = torch.cat(input_images,0)

        # Generates training targets for Panoptic-DeepLab.
        targets = self.panoptic_target_generator(input_panoptic_seg, select_segments_infos)
        tmp_list = targets['v_centerratiopoints']
        tmptmp = []
        for lll in tmp_list:
            tmptmp.extend(lll)
        if  len(tmp_list)==0:
            pass
        else:
            FLAG=1
    dataset_dict["video_images"] = input_images
    dataset_dict.update(targets)
        



    return    dataset_dict
        
