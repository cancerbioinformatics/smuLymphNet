
import os
import glob
import time
import random
import json
import random
import argparse
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

from network_gc import UNet_multi as msunet_gc
from network_sinus import UNet_multi as msunet_sinus
#from utils import *
from stitching import stitch, Canvas
from utilities import TissueDetect, ln_post_processing


def patching(dims, step, tile_dim):
    y_dim,x_dim=dims
    for y in range(0, y_dim, step):
        for x in range(0, x_dim, step):
            x_new = x_dim-tile_dim if x+tile_dim>x_dim else x
            y_new = y_dim-tile_dim if y+tile_dim>y_dim else y
            yield x_new, y_new


def get_transform(image):
    image=image.convert('RGB')
    transform = T.ToTensor()
    n = T.Normalize(
        (0.7486, 0.5743, 0.7222),
        (0.0126, 0.0712, 0.0168))
    image = transform(image)
    image = n(image)
    return image


def predict(model, tile, thold, args, feature="gc"):
    
    model.to(device)
    tile=(get_transform(tile).unsqueeze(0)).to(device)
    prediction=model(tile)
    probs=F.softmax(prediction,1)
                
    probs=(probs[0][1,:,:].cpu().detach().numpy())*255.0
    probs=probs.astype(np.uint8)
    new_dim=(args.tile_dim//args.downsample,args.tile_dim//args.downsample) 
    probs=cv2.resize(probs, new_dim, interpolation = cv2.INTER_AREA)
    
    label = 255 if feature=="gc" else 128
    probs[probs<=thold]=0
    probs[probs>thold]=label
    #probs[probs>255]=255
    return probs


def get_segmentation(slide,model_gc,model_sinus,args):

    ds = int(int(args.base_mag) / 10)
    ds_factors = [int(d) for d in slide.level_downsamples]
    level = ds_factors.index(ds)
    print(f'Downsample: {ds} \nLevel: {level}')

    margin=int((args.tile_dim-args.stride)/2)
    wsi_dims=slide.level_dimensions[level] 

    if 32 in ds_factors:
        canvas_level = ds_factors.index(32)
    else:
        canvas_level = ds_factors.index(ds_factors[-1])

    canvas_dims = slide.level_dimensions[canvas_level]
    args.downsample = int(wsi_dims[0] / canvas_dims[0])

    c=Canvas(canvas_dims[1], canvas_dims[0])
    print('Segmenting...')
    for x, y in patching(wsi_dims, args.stride, args.tile_dim):
        try:
            tile = slide.read_region((
                y*ds,x*ds),
                level,
                (args.tile_dim,args.tile_dim)
            )
        except openslide.lowlevel.OpenSlideError as e:
            print(e)
            return None
            
        if model_gc is not None:
            gc_thold=int(255*args.gc_threshold)
            gc_probs=predict(model_gc, tile, gc_thold, args, "gc")
        if model_sinus is not None:
            sinus_thold=int(255*args.sinus_threshold)
            sinus_probs=predict(model_sinus, tile, sinus_thold, args, "sinus")
        if (model_gc is not None) and (model_sinus is not None):
            probs=gc_probs.astype(np.float32)+sinus_probs.astype(np.float32)
            if max(probs.ravel().tolist())>255:
                probs[probs==383]=255
        elif (model_gc is None) and (model_sinus is not None):
            probs=sinus_probs
        elif (model_gc is not None) and (model_sinus is None):
            probs=gc_probs
        else:
            print('No models loaded')
        
        stitch(
            c,
            probs, 
            int(x//args.downsample), 
            int(y//args.downsample), 
            canvas_dims[0],#h,
            canvas_dims[1],#w
            int(args.tile_dim/args.downsample),
            int(args.stride/args.downsample), 
            int(margin/args.downsample)
        )

    return c.canvas  


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument(
        '-gm', 
        '--germinal_model',
        default=None,
        help='path to trained torch GC model'
    )
    ap.add_argument(
        '-sm', 
        '--sinus_model',
        default=None,
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
    #ap.add_argument(
        #'-bm',
        #'--base_level',
        #default="40",
        #help='WSI base magnification level'
    #)
    ap.add_argument(
        '-ts',
        '--tile_dim',
        default=1600,
        help='dimensions of tiles (tile_sizextile_size)'
    )
    ap.add_argument(
        '-ss',
        '--stride',
        default=600,
        help='size of stride for stepping across WSI'
    )
    ap.add_argument(
        '-ds',
        '--downsample',
        default=8,
        help='downsample size'
    )
    args=ap.parse_args()
    print('Initiating.....',flush=True)

    torch.set_grad_enabled(False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device2 =torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print("Loading models....",flush=True)

    if args.germinal_model is not None:
        model_gc = msunet_gc(3,2)
        model_gc.load_state_dict(torch.load(args.germinal_model,map_location='cpu'))
        print('loading GC model')
    else:
        model_gc=None

    if args.sinus_model is not None:
        model_sinus = msunet_sinus(3,2)
        model_sinus.load_state_dict(torch.load(args.sinus_model,map_location='cpu'))
        print('loading sinus model')
    else:
        model_sinus=None

    image_paths=glob.glob(os.path.join(args.wsi_path,'*'))[1:20]
    print('num images:{}'.format(len(image_paths)),flush=True)
    durations = []
    errors = []
    for i in range(len(image_paths)):
        image_path=image_paths[i]
        image_name=os.path.basename(image_path)
        slide=openslide.OpenSlide(image_path)
        print(f'Slide:{image_name}',flush=True)
        
        args.base_mag = slide.properties[
            openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        print(f'Base mag: {args.base_mag}')
        start = time.time()
        pred_mask=get_segmentation(slide,model_gc,model_sinus,args)
        if pred_mask is None:
            print(f'{image_name} is corrupted')
            errors.append(image_name)
            continue

        done = time.time()
        elapsed = done - start
        durations.append(elapsed)
        print(f'Prediction time: {elapsed}')
         
        td = TissueDetect(slide)
        contours = td._generate_tissue_contour()
        image = td.tissue_thumbnail
        
        mask_test = ln_post_processing(pred_mask, contours, labels=[128, 255])

        cv2.imwrite(os.path.join(args.save_path,image_name+'_predmask.png'),pred_mask) 
        cv2.imwrite(os.path.join(args.save_path,image_name+'_filteredmask.png'),mask_test) 
        cv2.imwrite(os.path.join(args.save_path,image_name+'_wsithumb.png'),image) 

    print(f'Finished segmenting {len(image_paths)} WSI')
    print(f'Average prediction duration: {np.mean(durations)}')

    err_df = pd.DataFrame({'errors': errors})
    err_df.to_csv(os.path.join(args.save_path,'errors.csv'))
