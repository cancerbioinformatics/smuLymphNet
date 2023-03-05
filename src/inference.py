#Amit Lohan

import os
import glob
import random
import json
import random
import datetime
from PIL import Image

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from network_gc import UNet_multi as UNet_gc
from network_sinus import UNet_multi as UNet_sinus
from utils import *


torch.set_grad_enabled(False)

def wsi_multiscale_inference(model_gc,model_sinus):

    try:
        image_dims=get_imdims(image_path)
    except Exception as e:
        print(e,flush=True)
         continue
    
    downsample_x=int(image_dims[1]/downsample_factor + 1)
    downsample_y=int(image_dims[0]/downsample_factor + 1)
    pred_mask=np.zeros((downsample_x,downsample_y))

    num_patches=get_num_patches(image_path,None,chopsize,slidesize)
    slide=openslide.OpenSlide(image_path)
    #slide_thumbnail=slide.get_thumbnail((1000,1000))
    #slide_thumbnail=np.array(slide_thumbnail)   
    #thumb_path=os.path.join(SAVE_PATH,image_name+'_thumbnail.png')
    #cv2.imwrite(thumb_path,slide_thumbnail)
 
    for patch,x,y,pos in get_patches(image_path,None,chopsize,slidesize):
        start_time = datetime.datetime.now()
        if pos!='m':
            continue
        
        #GERMINAL prediction

        model_gc.to(device)
        patch=(get_transform(patch).unsqueeze(0)).to(device)
        prediction_gc=model_gc(patch)
        model_gc=model_gc.cpu()
        prediction_gc.to(device)

        #SINUS prediction
        model_sinus.to(device)
        patch=patch.to(device)
        prediction_sinus=model_sinus(patch)
        model_sinus=model_sinus.cpu()
        prediction_sinus.to(device)

        #Class probabilities
        #prediction_gc.to(device)
        #prediction_sinus.to(device)
        probs_gc=F.softmax(prediction_gc,1)
        probs_sinus=F.softmax(prediction_sinus,1)
                
        parr_gc=(probs_gc[0][1,:,:].cpu().detach().numpy())*255.0
        parr_sinus=(probs_sinus[0][1,:,:].cpu().detach().numpy())*255.0
        parr_gc=parr_gc.astype(np.uint8)
        parr_sinus=parr_sinus.astype(np.uint8)

        parr_gc = cv2.resize(parr_gc, ddims, interpolation = cv2.INTER_AREA)
        parr_sinus = cv2.resize(parr_sinus, ddims, interpolation = cv2.INTER_AREA)
        
        #Get class prediction
        parr_gc[parr_gc<=gc_thold]=0
        parr_gc[parr_gc>gc_thold]=255
        parr_sinus[parr_sinus<=sinus_thold]=0
        parr_sinus[parr_sinus>sinus_thold]=128
            
        #parr=np.array(patch)[:,:,0]
        #parr=cv2.resize(parr,ddims, interpolation = cv2.INTER_AREA)
        x=x//downsample_factor
        y=y//downsample_factor 
        parr=parr_gc+parr_sinus
        parr[parr>255]=255
        #parr=parr.astype(np.uint8)

        #merge patches
        pred_mask=stitch_patch(
            pred_mask,
            parr,
            x,
            y,
            pos,
            slidesize//downsample_factor,
            chopsize//downsample_factor,
            margin//downsample_factor
            )
        #Added save to patient directory in save folder - Greg
    return pred_mask


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument(
        '-gm', 
        '--germinal_model',
        required=True,
        help='path to trained torch GC model'
    )

    ap.add_argument(
        '-sm', 
        '--sinus_model',
        required=True,
        help='path to trained torch sinus model'
    )

    ap.add_argument(
        '-wp',
        '--wsi_path',
        required=True,
        help='path to WSIs'
    )

    ap.add_argument(
        '-sp',
        '--save_path',
        required=True,
        help='directory to save results'
    )

    ap.add_argument(
        '-gt',
        '--gc_threshold',
        default=0.9,
        help='gc model prediction threshold'
    )

    ap.add_argument(
        '-st',
        '--sinus_threshold',
        default=0.9,
        help='sinus model prediction threshold'
    )

    ap.add_argument(
        '-bl',
        '--mag_base_level',
        default=0,
        help='WSI base magnification level'
    )

    args=parser.parse_args()
    print('Initiating.....',flush=True)

    torch.set_grad_enabled(False)

    downsample_factor=10
    chopsize=downsample_factor*160
    slidesize=600
    margin=int((chopsize-slidesize)/2)
    gc_thold=int(255*args.gc_threshold)
    sinus_thold=int(255*args.sinus_threshold)
    ddims=(chopsize//downsample_factor,chopsize//downsample_factor)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 =torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print("Loading models....",flush=True)
    model_gc = UNet_gc(3,2)
    model_gc.load_state_dict(torch.load(args.germinal_model,map_location='cpu'))
    model_gc.to(device)
    model_sinus = UNet_sinus(3,2)
    model_sinus.load_state_dict(torch.load(args.sinus_model,map_location='cpu'))
    model_sinus.to(device2)

    image_paths=glob.glob(os.path.join(args.wsi_path,'*'))
    print('num images:{}'.format(len(image_paths)),flush=True)

    for i in range(len(image_paths)):
        image_path=image_paths[i]
        image_name=os.path.basename(image_path)
        print(f'Slide:{image_name}',flush=True)
        pred_mask=wsi_multiscale_inference(image_path,model_gc,model_sinus)
        cv2.imwrite(os.path.join(args.save_path,image_name+'.png'),pred_mask) 

    print(f'Finished segmenting {len(image_paths)} WSI')
