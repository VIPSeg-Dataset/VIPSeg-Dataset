# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager



def load_video_vspw_vps_json(json_file, image_dir, gt_dir):
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

        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    videoid_img_dic={}
    for video_ in json_info["videos"]:
        videoid_img_dic[video_['video_id']]={}
        for imgimg in video_['images']:
        
            videoid_img_dic[video_['video_id']][imgimg['id']]={'width':imgimg['width'],'height':imgimg['height'],'file_name':imgimg['file_name']}

    cate_id_thingstaff={}
    for cate in json_info['categories']:
        cate_id_thingstaff[cate['id']] = cate['isthing']

    ret = []
    for ann in json_info["annotations"]:
        video_id = ann["video_id"]
        anns = ann['annotations']        
        image_files = []
        label_files = []
        sem_label_files = []
        segments_infos = []
        for image in anns:
            image_id = image['image_id']
      
        
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
            image_file = os.path.join(image_dir,video_id, videoid_img_dic[video_id][image_id]['file_name'].split('.')[0]+'.jpg')
            image_files.append(image_file)
            
            label_file = os.path.join(gt_dir, video_id,image["file_name"])
            label_files.append(label_file)

#            sem_label_file = os.path.join(semseg_dir, image["file_name"])
#            sem_label_files.append(sem_label_file)

            segments_info = image["segments_info"]
            segments_info = [_convert_category_id(seg_info,cate_id_thingstaff) for seg_info in segments_info] 
            segments_infos.append(segments_info)
        ret.append(
            {
                "file_names": image_files,
                "width": videoid_img_dic[video_id][image_id]['width'],
                "height": videoid_img_dic[video_id][image_id]['height'],
                "video_id": video_id,
                "pan_seg_file_names": label_files,
        #        "sem_seg_file_names": sem_label_files,
                "segments_infos": segments_infos,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
#    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
#    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
#    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    #print(ret[0])
    #exit()
    return ret


def register_video_vspw_vps_json(
    name, metadata, image_root, panoptic_root,  panoptic_json, instances_json=None
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
        lambda: load_video_vspw_vps_json(
            panoptic_json, image_root, panoptic_root
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type=None,
        ignore_label=255,
        label_divisor=100,
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

    
    thing_classes = [k["name"] for k in json_info['categories'] if k['isthing']]
    thing_colors = [k["color"] for k in  json_info['categories'] if k['isthing']]
    stuff_classes = [k["name"] for k in json_info['categories'] if not k['isthing']]
    stuff_colors = [k["color"] for k in json_info['categories'] if not k['isthing']]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors


    thing_classes_id = [k['id']  for k in json_info['categories'] if k['isthing']]
    meta['thing_classes_id'] = thing_classes_id

    meta['categories'] = json_info['categories']

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


_PREDEFINED_SPLITS_PANOVSPW = {
    "panoVSPW_vps_video_train": (
        "panoVSPW_gts/panoVSPW_720p/images",
        "panoVSPW_gts/panoVSPW_720p/panomasksRGB",
        "panoVSPW_gts/panoVSPW_720p/panoptic_gt_vspw_train.json",
    ),
    "panoVSPW_vps_video_val": (
        "panoVSPW_gts/panoVSPW_720p/images",
        "panoVSPW_gts/panoVSPW_720p/panomasksRGB",
        "panoVSPW_gts/panoVSPW_720p/panoptic_gt_vspw_val.json",
    ),
    "panoVSPW_vps_video_test": (
        "panoVSPW_gts/panoVSPW_720p/images",
        "panoVSPW_gts/panoVSPW_720p/panomasksRGB",
        "panoVSPW_gts/panoVSPW_720p/panoptic_gt_vspw_test.json",
    ),
    }


def register_all_video_panoVSPW(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json),
    ) in _PREDEFINED_SPLITS_PANOVSPW.items():
        metadata = get_metadata(os.path.join(root, panoptic_json))
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_video_vspw_vps_json(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
#            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_video_panoVSPW(_root)

#register_all_ade20k_panoptic(_root)
