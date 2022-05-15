import os
import random
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import math


class VideoClipTestDataset:
    def __init__(self,list_ ,train_clipnum):
        self.list_ = list_
        self.train_clipnum = train_clipnum
    
    def __len__(self):
        if len(self.list_)>self.train_clipnum:                                                                  
            return len(self.list_) - self.train_clipnum+1                                                       
        else:                                                                                                   
            return 1  

    def getdata(self,idx):
        
        out = {}
        imgnames = []
        imgs= []
         
        if len(self.list_)>self.train_clipnum:                                                                  
            ix_list = list(range(idx, idx+self.train_clipnum))                                                  
        else:                                                                                                   
            ix_list = list(range(len(self.list_)))                                                              
        for ii in ix_list:  
            img_dir = self.list_[ii]
            imgname = img_dir.split('/')[-1]
            img = Image.open(img_dir) 
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.permute((2, 0, 1)).contiguous()
            _,h,w = img.size()
            imgs.append(img.unsqueeze(0))
            imgnames.append(imgname)
        imgs = torch.cat(imgs,0)
        out['video_images'] = imgs
        out['image_names'] = imgnames
        out['height'] = h
        out['width'] = w
        return out
        
class VideoClipTestNooverlapDataset:
    def __init__(self,list_ ,train_clipnum):
        self.list_ = list_
        self.train_clipnum = train_clipnum
    
    def __len__(self):
        if len(self.list_)>self.train_clipnum:                                                                  
            return math.ceil(len(self.list_)/self.train_clipnum)                                                      
        else:                                                                                                   
            return 1  

    def getdata(self,idx):
        
        out = {}
        imgnames = []
        imgs= []
         
        if len(self.list_)>self.train_clipnum:                                                                  
            ix_list = list(range(idx*self.train_clipnum, (idx+1)*self.train_clipnum))                                                  
        else:                                                                                                   
            ix_list = list(range(len(self.list_)))                                                              
        for ii in ix_list:  
            if ii>=len(self.list_):
                continue
            img_dir = self.list_[ii]
            imgname = img_dir.split('/')[-1]
            img = Image.open(img_dir) 
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.permute((2, 0, 1)).contiguous()
            _,h,w = img.size()
            imgs.append(img.unsqueeze(0))
            imgnames.append(imgname)
        imgs = torch.cat(imgs,0)
        out['video_images'] = imgs
        out['image_names'] = imgnames
        out['height'] = h
        out['width'] = w
        return out
            
            
        
        
        
