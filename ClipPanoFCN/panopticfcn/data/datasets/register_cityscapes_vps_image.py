# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager



def load_image_cityscapes_vps_json(json_file, image_dir, gt_dir, semseg_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, cate_id_thingstaff):
        isthing = cate_id_thingstaff[segment_info['category_id']]
        segment_info["isthing"] = isthing
#        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
#            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
#                segment_info["category_id"]
#            ]
#            segment_info["isthing"] = True
#        else:
#            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
#                segment_info["category_id"]
#            ]
#            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    imgid_img_dic={}
    for imgimg in json_info["images"]:
        imgid_img_dic[imgimg['id']]={'width':imgimg['width'],'height':imgimg['height'],'file_name':imgimg['file_name']}

    cate_id_thingstaff={}
    for cate in json_info['categories']:
        cate_id_thingstaff[cate['id']] = cate['isthing']

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, imgid_img_dic[image_id]['file_name'])
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = ann["segments_info"]
        segments_info = [_convert_category_id(seg_info,cate_id_thingstaff) for seg_info in segments_info] 
        ret.append(
            {
                "file_name": image_file,
                "width": imgid_img_dic[image_id]['width'],
                "height": imgid_img_dic[image_id]['height'],
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_image_cityscapes_vps_json(
    name, metadata, image_root, panoptic_root, semantic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_image_cityscapes_vps_json(
            panoptic_json, image_root, panoptic_root, semantic_root
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type='cityscapes_panoptic_seg',
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )




def get_metadata(json_file):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    
    #thing_classes = [k["name"] for k in json_info['categories'] if k['isthing']]
    #thing_colors = [k["color"] for k in  json_info['categories'] if k['isthing']]
    #stuff_classes = [k["name"] for k in json_info['categories'] if not k['isthing']]
    #stuff_colors = [k["color"] for k in json_info['categories'] if not k['isthing']]
    thing_classes = [k["name"] for k in json_info['categories'] ]
    thing_colors = [k["color"] for k in  json_info['categories'] ]
    stuff_classes = [k["name"] for k in json_info['categories'] ]
    stuff_colors = [k["color"] for k in json_info['categories'] ]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    thing_classes_id = [k['id']  for k in json_info['categories'] if k['isthing']]
    stuff_classes_id = [k['id']  for k in json_info['categories'] if not k['isthing']]
    for id_ in thing_classes_id:
        thing_dataset_id_to_contiguous_id[id_] = id_
    for id_ in stuff_classes_id:
        stuff_dataset_id_to_contiguous_id[id_] = id_
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.

        # in order to use sem_seg evaluator

    return meta


_PREDEFINED_SPLITS_CITYSCAPES_VPS = {
    "cityscapes_vps_image_train": (
        "cityscapes_vps/train/img",
        "cityscapes_vps/train/panoptic_video",
        "cityscapes_vps/image_panoptic_gt_train_city_vps.json",
        "cityscapes_vps/train/labelmap",
    ),
    "cityscapes_vps_image_val": (
        "cityscapes_vps/val/img",
        "cityscapes_vps/val/panoptic_video",
        "cityscapes_vps/image_panoptic_gt_val_city_vps.json",
        "cityscapes_vps/val/labelmap",
    ),
}


def register_all_image_cityscapes_vps(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_CITYSCAPES_VPS.items():
        metadata = get_metadata(os.path.join(root, panoptic_json))
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_image_cityscapes_vps_json(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_image_cityscapes_vps(_root)

#register_all_ade20k_panoptic(_root)
