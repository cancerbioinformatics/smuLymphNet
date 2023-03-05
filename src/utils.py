import numpy as np
import cv2
import openslide

from torchvision import transforms as T

LEVEL=1

def get_transform(img):
    img=img.convert('RGB')
    tt = T.ToTensor()
    n = T.Normalize(
        (0.7486, 0.5743, 0.7222),
        (0.0126, 0.0712, 0.0168))
    img = tt(img)
    img = n(img)
    return img

def get_iou(output_mask,gt_mask):
    output_mask=output_mask/255
    gt_mask=gt_mask/255
    num=np.sum((output_mask*gt_mask)*1.0)
    den=output_mask+gt_mask
    den[den==2]=1
    #print(np.unique(den))
    den=np.sum(den)
    if float(den)==0:
       return 1
    return float(num/(float(den)))

def stitch_patch(mask,parr,x,y,pos,slidesize,chopsize,margin):
    #print(x,y)
    #print("Initial Shape:",parr.shape,pos)
    if pos=='m':
       parr=parr[margin:margin+slidesize,margin:margin+slidesize]
       x+=margin
       y+=margin
    elif pos=='t':
       parr=parr[0:margin+slidesize,margin:margin+slidesize]
       y+=margin
    elif pos=='b':
       parr=parr[margin:chopsize,margin:margin+slidesize]
       x+=margin
       y+=margin
    elif pos=='l':
       parr=parr[margin:margin+slidesize,0:margin+slidesize]
       x+=margin
    elif pos=='r':
       parr=parr[margin:margin+slidesize,margin:chopsize]
       x+=margin
       y+=margin
    elif pos=='tl':
       parr=parr[0:margin+slidesize,0:margin+slidesize]
    elif pos=='tr':
       parr=parr[0:margin+slidesize,margin:chopsize]
       y+=margin
    elif pos=='ll':
       parr=parr[margin:chopsize,0:margin+slidesize]
       x+=margin
    elif pos=='lr':
       parr=parr[margin:chopsize,margin:chopsize]
       x+=margin
       y+=margin
    else:
       assert False,"Type of patch is unknown."
    #print("Inside stitcher:",parr.shape,x,y,x+parr.shape[0]-x,y+parr.shape[1]-y)
    #print(mask.shape)
    mask[x:x+parr.shape[0],y:y+parr.shape[1]]=parr
    return mask

def get_patches(img_path,mask_path,chopsize,slidesize):
    slide = openslide.OpenSlide(img_path)
    height,width=slide.level_dimensions[LEVEL]
    #print("Height:",height)
    x=0
    y=0    
    while y<height:
        while x<width:
            if x+chopsize>width:
                xtemp=width-chopsize
            else:
                xtemp=x
            if y+chopsize>height:
                ytemp=height-chopsize
            else:
                ytemp=y
            if ytemp==0:
               flag='l'
            elif ytemp==height-chopsize:
               flag='r'
            elif xtemp==0:
               flag='t'
            elif xtemp==width-chopsize:
               flag='b'
            else:
               flag='m'
               
            if ytemp==0 and xtemp==0:
               flag='tl'
            elif ytemp==height-chopsize and xtemp==0:
               flag='tr'
            elif ytemp==0 and xtemp==width-chopsize:
               flag='ll'
            elif ytemp==height-chopsize and xtemp==width-chopsize:
               flag='lr'
               
            yield slide.read_region((ytemp*2**LEVEL,xtemp*2**LEVEL),LEVEL,(chopsize,chopsize)),xtemp,ytemp,flag
            x+=slidesize
        y+=slidesize
        x=0

def get_num_patches(img_path,mask_path,chopsize,slidesize):

    slide = openslide.OpenSlide(img_path)
    im_dims=slide.level_dimensions[LEVEL] 
    width,height=im_dims[0],im_dims[1]
    num_patches=(width//slidesize+1)*(height//slidesize+1)
    return num_patches

def get_total_patches(images,chopsize,slidesize):
    t_p=0
    for img in images:
        t_p+=get_num_patches(img,None,chopsize,slidesize)
    return t_p

def get_imdims(img_path):
    slide = openslide.OpenSlide(img_path)
    im_dims=slide.level_dimensions[LEVEL] 
    return im_dims
