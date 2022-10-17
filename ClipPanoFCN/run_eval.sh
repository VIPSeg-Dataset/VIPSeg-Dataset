export DETECTRON2_DATASETS='/your/path/to/VIPSeg_720P'

GPU_NUM=4
DATAROOT=$DETECTRON2_DATASETS'/VIPSeg/VIPSeg_720P/panomasksRGB'
PRETRAIN_WEIGHT='./panoptic_fcn_star_r50_3x.pth'

BATCHSIZE=8
CLIPNUM=3

CONFIG='configs/video_vpsw_PanopticFCN-R50-3x.yaml'

VAL_CLIPNUM=$CLIPNUM
IMGSAVEROOT='imgsave_f2'
SPLIT='val'
GT_JSONFILE=$DETECTRON2_DATASETS'/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_'$SPLIT'.json'

python eval_net_video.py --num-gpus 1 \
 --config-file $CONFIG  --pretrain_weight $PRETRAIN_WEIGHT --train_clipnum $VAL_CLIPNUM  --img_per_batch $BATCHSIZE --imgsaveroot $IMGSAVEROOT --split $SPLIT
###VPQ
python eval_vpq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
###STQ
python eval_stq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
