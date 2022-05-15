export DETECTRON2_DATASETS='/your/path/to/VIPSeg_720P'

GPU_NUM=4
PRETRAIN_WEIGHT='./panoptic_fcn_star_r50_3x.pth'

BATCHSIZE=8
CLIPNUM=3

CONFIG='configs/video_vpsw_PanopticFCN-R50-3x.yaml'


python train.py --config-file $CONFIG --num-gpus $GPU_NUM --pretrain_weight $PRETRAIN_WEIGHT --train_clipnum $CLIPNUM --img_per_batch $BATCHSIZE

