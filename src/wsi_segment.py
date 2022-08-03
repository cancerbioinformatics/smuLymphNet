import torch
import torch.nn.functional as F
import torch.nn as nn
from network_gc import UNet_multi as UNet_gc
from network_sinus import UNet_multi as UNet_sinus
from lutils import get_imdims,stitch_patch,get_patches,get_num_patches,get_total_patches,get_iou
from torchvision import transforms as T
from PIL import Image
import random
import json
import os
import random
import datetime
import cv2
import numpy as np
import glob
import openslide
import pandas as pd
#import sp_mask

torch.set_grad_enabled(False)

print("Initiating.....",flush=True)

def mine(d):
    if not os.path.exists(d):
        os.makedirs(d)

downsample_factor=10

cn='sinus_gray'
chopsize=downsample_factor*160
slidesize=600
margin=int((chopsize-slidesize)/2)
#margin=50
gc_thold=int(255*0.98)
sinus_thold=int(255*0.7)

ddims=(chopsize//downsample_factor,chopsize//downsample_factor)

df_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/multiscale-testing/data/guysln_test_set1.csv"
im_root="/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/wsi/train"
gc_model_path="/home/verghese/lymphnode-keras/multiscale-testing/models/gc.pth"
sinus_model_path="/home/verghese/lymphnode-keras/multiscale-testing/models/sinus.pth"
save_root="/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/multiscale-testing/guysln-test-set-0.99"
pred_root=save_root

mine(pred_root)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device2 =torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_transform(img):
    #img = Image.fromarray(img)
    img=img.convert('RGB')
    tt = T.ToTensor()
    n = T.Normalize((0.7486, 0.5743, 0.7222),(0.0126, 0.0712, 0.0168))
    img = tt(img)
    img = n(img)
    return img

print("Loading models....",flush=True)
model_gc = UNet_gc(3,2)
model_gc.load_state_dict(torch.load(gc_model_path,map_location='cpu'))
model_gc.to(device)
model_sinus = UNet_sinus(3,2)
model_sinus.load_state_dict(torch.load(sinus_model_path,map_location='cpu'))
model_sinus.to(device2)

im_num=1
total_processed=0
delta_sum=(datetime.datetime.now()-datetime.datetime.now())
delta_ctr=0

df=pd.read_csv(df_path)
image_paths=list(df['paths'])
images=list(df['images'])
patients=list(df['patients'])
print('num images:{}'.format(len(image_paths)),flush=True)

for i in range(len(image_paths)):
    im_path=image_paths[i]
    im_name=images[i]
    patient_id=str(patients[i])
    idx=patient_id.split('.')[0]

    print('idx:{},patient:{},image_name,:{},'.format(idx,patient_id,im_name),flush=True)
    print('path:{}'.format(im_path))
    os.makedirs(os.path.join(save_root,patient_id),exist_ok=True)

    try:
        im_dims=get_imdims(im_path)
    except Exception as e:
        print(e,flush=True)
        continue

    pred_mask=np.zeros((int(im_dims[1]/downsample_factor + 1),int(im_dims[0]/downsample_factor + 1)))

    num_patches=get_num_patches(im_path,None,chopsize,slidesize)
    patch_num=1
    slide=openslide.OpenSlide(im_path)
    slide_thumbnail=slide.get_thumbnail((1000,1000))
    slide_thumbnail=np.array(slide_thumbnail)
    cv2.imwrite(os.path.join(pred_root,patient_id,im_name+'_thumbnail.png'),slide_thumbnail)

    
    for patch,x,y,pos in get_patches(im_path,None,chopsize,slidesize):
        start_time = datetime.datetime.now()
        #patch.save('./out_patches/im'+str(patch_num)+'.png')
        print('greg')
        if pos!='m':
           total_processed+=1
           patch_num+=1
           continue
        print('greg2')
        patch=(get_transform(patch).unsqueeze(0)).to(device)

        
        prediction_gc=model_gc(patch)

        patch=patch.to(device2)
        prediction_sinus=model_sinus(patch)
        prediction_gc.to(device)
        prediction_sinus.to(device)

        probs_gc=F.softmax(prediction_gc,1)
        probs_sinus=F.softmax(prediction_sinus,1)
        
        print('gggggg',probs_gc.shape)
        parr_gc=(probs_gc[0][1,:,:].cpu().detach().numpy())*255.0
        parr_sinus=(probs_sinus[0][1,:,:].cpu().detach().numpy())*255.0
        #parr_gc=parr_gc.astype(np.uint8)
        #parr_sinus=parr_sinus.astype(np.uint8)

        parr_gc = cv2.resize(parr_gc, ddims, interpolation = cv2.INTER_AREA)
        parr_sinus = cv2.resize(parr_sinus, ddims, interpolation = cv2.INTER_AREA)

        parr_gc[parr_gc<=gc_thold]=0
        parr_gc[parr_gc>gc_thold]=255

        parr_sinus[parr_sinus<=sinus_thold]=0
        parr_sinus[parr_sinus>sinus_thold]=128

        x=x//downsample_factor
        y=y//downsample_factor

        parr=parr_gc+parr_sinus
        #parr = parr_gc

        parr[parr>255]=255

        parr=parr.astype(np.uint8)

        pred_mask=stitch_patch(pred_mask,parr,x,y,pos,slidesize//downsample_factor,chopsize//downsample_factor,margin//downsample_factor)
        total_processed+=1
        end_time = datetime.datetime.now()

        delta_sum+=(end_time-start_time)
        delta_ctr+=1

        #ETA=str((total_patches-total_processed)*(delta_sum/delta_ctr)).split('.')[0]
        #print("Processed patch",patch_num,"of",num_patches,"  Image file ",im_num,'of',num_ims,"  ",round(float(total_processed*100)/total_patches,2),"% Completed"," ETA:",ETA)
        patch_num+=1
        #print("Processed patch",patch_num,"of",num_patches,"  Imagefile",im_num,'of',num_ims)
    #Added save to patient directory in save folder - Greg
    print(os.path.join(pred_root,patient_id,im_name+'.png'),flush=True)
    cv2.imwrite(os.path.join(pred_root,patient_id,im_name+'.png'),pred_mask) 
    

#config_dict={'model':'gc_model_path', 'datetime':str(datetime.datetime.now())}
#with open(os.path.join(pred_root,'sample.json', 'w') as outfile:
#    json.dump(config_dict, outfile)

