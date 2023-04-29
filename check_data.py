from PIL import Image
import os

#dir_ = '/home/miaojiaxu/jiaxu_2/semantic_seg/panoptic_label/final_refined_manage'
#dir_='/home/miaojiaxu/jiaxu_2/semantic_seg/panoptic_label/final_all_pano_VSPW'
#dir_='final_all_pano_VIPSeg'
#dir_='/home/miaojiaxu/jiaxu_2/semantic_seg/panoptic_label/VIPSeg_gts/VIPSeg_720p/images'
#dir_='/home/miaojiaxu/jiaxu_2/semantic_seg/panoptic_label/VIPSeg_gts/VIPSeg_720p/panomasks'
#dir_='VIPSeg_720P/panomasks'
#dir_='VIPSeg_720P/images'
dir_='panomasks'
#dir_='imgs'
count=0
list_=[]
for video in os.listdir(dir_):
    for imgname in os.listdir(os.path.join(dir_,video)):
        count+=1
        if count%5000==0:
            print(count)
        #print(imgname)
        img = Image.open(os.path.join(dir_,video,imgname))
        w,h = img.size
        list_.append(w/h)
        if h>0 and w>0:
            continue
        else:
            print("video: {} image:{}".format(video,imgname))
print(count)
print(min(list_))
print(max(list_))
