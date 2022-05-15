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
from detectron2.structures import BitMasks, Instances,Boxes
from detectron2.projects.point_rend import ColorAugSSDTransform
from panopticapi.utils import rgb2id

from panopticfcn.target_generator import VideoPanopticDeepLabTargetGenerator

from .utils import Video_BitMasks,Video_Boxes

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
        thing_ids_to_continue_dic,
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
        self.thing_ids_to_continue_dic = thing_ids_to_continue_dic
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
                    1.0,
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


        #######
        thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values())
        thing_ids_to_continue_dic = {}
        for ii, id_ in enumerate(sorted(thing_ids)):
            thing_ids_to_continue_dic[id_] = ii

        #######

        panoptic_target_generator = VideoPanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            stuff_ids=list(meta.stuff_dataset_id_to_contiguous_id.values()),
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
            "train_clipnum": cfg.MODEL.TRAIN_CLIPNUM,
             "panoptic_target_generator":panoptic_target_generator,
            "thing_ids_to_continue_dic":thing_ids_to_continue_dic,
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

        #print(dataset_dict)
        video_length = len(dataset_dict['file_names'])
#        assert self.train_clipnum < video_length
        if self.train_clipnum < video_length:
            choice_index = np.random.choice(video_length-self.train_clipnum,1)        
            choice_index = int(choice_index[0])
            index_list = list(range(choice_index,choice_index+self.train_clipnum))
        else:
            index_list = list(range(video_length))
            while len(index_list)<self.train_clipnum:
                index_list.append(index_list[-1])
        
        select_filenames = []
        #select_sem_seg_file_names = [] 
        select_pan_seg_file_names = []
        select_segments_infos = []
         
         
        for idx in index_list:
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

        insid_catid_dic={}
        input_images=[]
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
        dataset_dict["video_images"] = input_images

        # Generates training targets for Panoptic-DeepLab.
        targets = self.panoptic_target_generator(input_panoptic_seg, select_segments_infos)
        dataset_dict.update(targets)


        image_shape = (input_images.shape[-2], input_images.shape[-1])
        input_panoptic_seg = np.stack(input_panoptic_seg)
        unique_ids = np.unique(input_panoptic_seg)

        instances = Instances(image_shape)
        classes = []
        masks = []
        bboxes = []
        for insid_,class_id_ in insid_catid_dic.items():
            if class_id_ not in self.thing_ids_to_continue_dic:
                continue

            if insid_ in unique_ids:
                classes.append(self.thing_ids_to_continue_dic[class_id_])
                masks.append(input_panoptic_seg == insid_)

              
                masks_ = np.array(input_panoptic_seg == insid_)
                #print(masks_.shape)
                bboxes_ = []
                for ii_ in range(len(masks_)):
                    mask_ = masks_[ii_]
                
                    xx,yy =np.nonzero(mask_)

                    if len(xx)==0 or len(yy) == 0:
  
                        bboxes_.append([0,0,0,0])
                    else:
                        bboxes_.append([xx.min(),yy.min(),xx.max(),yy.max()])
                bboxes.append(bboxes_)

        classes = np.array(classes)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            pass
        else:
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            bbox  = torch.tensor(np.array(bboxes))
            bbox = Video_Boxes(bbox)
            masks = Video_BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks
            instances.gt_boxes = bbox

        dataset_dict["instances"] = instances

        return    dataset_dict
        
