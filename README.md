# CVPR 2022: Large-scale Video Panoptic Segmentation in the Wild: A Benchmark


![avatar](show_data.png)


## VIPSeg DataSet: A large-scale VIdeo Panoptic Segmentation dataset. 

*The download links of VIPSeg*

```
Google Drive:
```

```
Baidu YunPan: https://pan.baidu.com/s/1rEO-G-T19Xh9ENni2s6_Zw  密码: 9ir3

```

## Instruction

The dataset is organized as following:

![avatar](org1.png =500x500)


*NOTE: For panoptic masks in *panomask/*, the IDs of categories are from 0 to 124. "0" denotes the VOID class. For "stuff" classes, the value of masks is the same as the category ID. For "thing" classes, the value of masks  is "category_id *100+instance_id". For instance, the category ID of "person" is 61. Then values of masks of the "person" instances are "6100","6101",... Thus, values of masks larger than 124 are belonging to things, otherwise it is stuff.*


### Change VIPSeg to 720P and COCO Format

```
python change2_720p.py

python create_panoptic_video_labels.py

python splitjson.py

```

The COCO format dataset is organized as following:


```
NOTE: 
```

## Baseline

```
cd ClipPanoFCN

sh run_train.sh

sh run_eval.sh

```


## Citation

```
@inproceedings{miao2022large,

  title={Large-scale Video Panoptic Segmentation in the Wild: A Benchmark},

  author={Miao, Jiaxu and Wang, Xiaohan and  Wu, Yu and Li, Wei and Zhang, Xu and Wei, Yunchao and Yang, Yi},

  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},

  year={2022}

}
```












