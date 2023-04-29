import os
import json

#for split in ['train','val','test']:
for split in ['train','val']:
    with open(split+'.txt','r') as f:
        lines = f.readlines()
        v_list = [line[:-1] for line in lines]

    with open('VIPSeg_720P/panoptic_gt_VIPSeg.json','r') as f:
        dic = json.load(f)
    dic_new ={}
    for key,value in dic.items():
        if key=='categories':
            dic_new[key] = value
        else:
            list_=[]
            for v in value:
                if v['video_id'] in v_list:
                    list_.append(v)
            dic_new[key] = list_
    with open('VIPSeg_720P/panoptic_gt_VIPSeg_'+split+'.json','w') as f:
        json.dump(dic_new,f)
