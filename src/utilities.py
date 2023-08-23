
import os 
import glob
from itertools import chain

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl  
import matplotlib.patches as patches
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import square, closing, opening


def within_contour(pt, contours, label=1):
    target=0
    for c in contours:
        if cv2.pointPolygonTest(c, pt, False)==1:
            target=label
            break
    return target


def ln_post_processing(seg_mask, contours, labels):
    
    mask_filtered = np.zeros((seg_mask.shape))
    for l in labels:
        mask=seg_mask.copy()
        mask[mask==l]=1
        coords=list(zip(*np.where(mask==1)))
        #mask=cv2.resize(mask,tuple(reversed(slide_adj.shape[0:2])))
        
        for i, j in coords:
            new_label=within_contour((int(j),int(i)),contours,l)
            mask[i,j]=new_label

        mask_filtered[mask==l]=l
                                 
    return mask_filtered 


class TissueDetect():

    bilateral_args=[
            {"d":9,"sigmaColor":10000,"sigmaSpace":150},
            {"d":90,"sigmaColor":5000,"sigmaSpace":5000},
            {"d":90,"sigmaColor":10000,"sigmaSpace":10000},
            {"d":90,"sigmaColor":10000,"sigmaSpace":100}
            ]

    thresh_args=[
            {"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU},
            {"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}
            ]

    def __init__(self, slide):
        self.slide=openslide.OpenSlide(slide) if isinstance(slide, str) else slide
        self.tissue_mask=None 
        self.contour_mask=None
        self._border=None


    @property
    def tissue_thumbnail(self):

        ds = [int(d) for d in self.slide.level_downsamples]
        level = ds.index(32) if 32 in ds else ds.index(int(ds[-1]))
        contours=self._generate_tissue_contour()
        image=self.slide.get_thumbnail(self.slide.level_dimensions[level])
        image=np.array(image.convert('RGB'))
        cv2.drawContours(image, contours, -1, (0, 255, 0), 5)
        #x,y,w,h=cv2.boundingRect(np.concatenate(contours))
        #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)

        return image
        

    def border(self,mag_level):

        test=cv2.resize(self.contour_mask,self.slide.dimensions)
        contours,_=cv2.findContours(test,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        #test=cv2.resize(self.contour_mask,self.slide.dimensions)
        #image=self.slide.get_thumbnail(self.slide.level_dimensions[3])
        #image=np.array(image.convert('RGB'))
        #contour=contours[np.argmax([c.size for c in contours])]
        x,y,w,h=cv2.boundingRect(np.concatenate(contours))
        #x,y,w,h=[d*int(self.slide.level_downsamples[mag_level]) for d in [x,y,w,h]]
        self._border=((x,y),(x+w,y+h))
        return self._border
        

    def detect_tissue(self,mask_level):
    
        image = self.slide.read_region((0,0),mask_level, 
                    self.slide.level_dimensions[mask_level]) 

        image = self.slide.get_thumbnail(self.slide.level_dimensions[mask_level]) 
        image = np.array(image.convert('RGB'))
        gray = rgb2gray(image)
        gray_f = gray.flatten()

        pixels_int = gray_f[np.logical_and(gray_f > 0.1, gray_f < 0.98)]
        t = threshold_otsu(pixels_int)
        thresh = np.logical_and(gray_f<t, gray_f>0.1).reshape(gray.shape)
        
        mask = opening(closing(thresh, selem=square(2)), selem=square(2))
        self.tissue_mask = mask.astype(np.uint8)
        
        return cv2.resize(mask.astype(np.uint8),self.slide.dimensions)


    def _generate_tissue_contour(self):
        
        ds = [int(d) for d in self.slide.level_downsamples]
        level = ds.index(32) if 32 in ds else ds.index(int(ds[-1]))
        slide=self.slide.get_thumbnail(self.slide.level_dimensions[level])
        slide=np.array(slide.convert('RGB'))
        img_hsv=cv2.cvtColor(slide,cv2.COLOR_RGB2HSV)
        lower_red=np.array([120,0,0])
        upper_red=np.array([180,255,255])
        mask=cv2.inRange(img_hsv,lower_red,upper_red)
        img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
        m=cv2.bitwise_and(slide,slide,mask=mask)
        im_fill=np.where(m==0,233,m)
        mask=np.zeros(slide.shape)
        gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)
        
        for b in TissueDetect.bilateral_args:
            gray=cv2.bilateralFilter(np.bitwise_not(gray),**b)
        blur=255-gray
        
        for t in TissueDetect.thresh_args:
            _,blur=cv2.threshold(blur,**t)
        
        self.contour_mask=blur
        contours,_=cv2.findContours(blur,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours=list(filter(lambda x: cv2.contourArea(x) > 5000, contours))
        self.contours=contours
        return self.contours
