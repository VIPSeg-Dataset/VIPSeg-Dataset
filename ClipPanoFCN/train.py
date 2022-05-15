# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PanopticFCN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from panopticfcn import add_panopticfcn_config, build_lr_scheduler
os.environ["NCCL_LL_THRESHOLD"] = "0"
from detectron2.data import MetadataCatalog,build_detection_train_loader
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from panopticfcn import (
#    PanopticDeeplabDatasetMapper,
#    add_panoptic_deeplab_config,
    PanopticDatasetVideoMapper,
    PanopticDatasetImageMapper,
    PanopticDatasetVideoTwoframeMapper,)

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
        elif cfg.INPUT.DATASET_MAPPER_NAME == "video_mapper":
            mapper = PanopticDatasetVideoMapper(cfg,True)
        else:
            pass
#            mapper = PanopticDatasetVideoMapper(cfg,True)
#        elif cfg.INPUT.DATASET_MAPPER_NAME =='video_twoframe':
#
#            mapper = PanopticDatasetVideoTwoframeMapper(cfg,True)
#        else:
#            mapper = PanopticDatasetImageMapper(cfg,True)
#            mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)


    cfg.defrost()
    #cfg.MODEL.WEIGHTS= os.path.join( args.pretrain_weight,'R-52.pkl')
    #cfg.MODEL.WEIGHTS= os.path.join( args.pretrain_weight,'model_final_coco.pkl')
    cfg.MODEL.WEIGHTS= os.path.join( args.pretrain_weight)
    cfg.SOLVER.IMS_PER_BATCH = args.img_per_batch
    cfg.MODEL.TRAIN_CLIPNUM = args.train_clipnum
#    cfg.SOLVER.FIX_BN = args.fix_bn
#    cfg.SOLVER.FIX_STAGEONE = args.fix_stageone
#    cfg.SOLVER.TRIPLET = args.triplet
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = default_argument_parser()
    parser.add_argument(
        "--pretrain_weight",type=str)
    parser.add_argument(
        "--train_clipnum",type=int)
    parser.add_argument(
        "--img_per_batch",type=int)
    args =  parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
