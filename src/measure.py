'''
measure.py: Slide, LymphNode, Germinal and Sinuses classes used to detect the predicted features and
quantify them using contouring methods
'''

import os
import glob

import cv2
import numpy as np


class Slide():

    bilateral1_args={"d":9,"sigmaColor":10000,"sigmaSpace":150}
    bilateral2_args={"d":90,"sigmaColor":5000,"sigmaSpace":5000}
    bilateral3_args={"d":90,"sigmaColor":10000,"sigmaSpace":10000}
    bilateral4_args={"d":90,"sigmaColor":10000,"sigmaSpace":100}
    thresh1_args={"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU}
    thresh2_args={"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}

    def __init__(
        self, 
        slide, 
        mask,
        resolution):
        #pixWidth=0.23e-6, 
        #pixHeight=0.23e-6):

        self.slide=slide
        self.mask=mask
        self.resolution=resolution
        #self.pixWidth=pixWidth
        #self.pixHeight=pixHeight
        self.contours=None
        self._lymphnodes=None

    @property
    def w_scale(self):
        #return (self.w/self.wNew)*self.pixWidth
        return self.resolution

    @property
    def h_scale(self):
        #return (self.h/self.hNew)*self.pixHeight
        return self.resolution

    def locate_nodes(self, germ_label, sinus_label):

        img_hsv=cv2.cvtColor(self.slide,cv2.COLOR_RGB2HSV)
        lower_red=np.array([120,0,0])
        upper_red=np.array([180,255,255])
        mask=cv2.inRange(img_hsv,lower_red,upper_red)
        img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
        m=cv2.bitwise_and(self.slide,self.slide,mask=mask)
        im_fill=np.where(m==0,233,m)
        mask=np.zeros(self.slide.shape)
        gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)
        blur1=cv2.bilateralFilter(np.bitwise_not(gray),**Slide.bilateral1_args)
        blur2=cv2.bilateralFilter(np.bitwise_not(blur1),**Slide.bilateral2_args)
        blur3=cv2.bilateralFilter(np.bitwise_not(blur2),**Slide.bilateral3_args)
        blur4=cv2.bilateralFilter(np.bitwise_not(blur3),**Slide.bilateral4_args)
        blur_final=255-blur4
        #threshold twice
        _,thresh=cv2.threshold(blur_final,**Slide.thresh1_args)
        _,thresh=cv2.threshold(thresh,**Slide.thresh2_args)
        #find contours
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours=list(filter(lambda x: cv2.contourArea(x) > 9000, contours))
        self.contours=contours
        self._lymphnodes=[self._create_node(c, thresh, germ_label, sinus_label) for c in contours]
        self._lymphnodes=list(filter(lambda x: x is not None, self._lymphnodes))
        return len(self._lymphnodes)


    def _create_node(self, contour, thresh, germ_label, sinus_label):

        x,y,w,h=cv2.boundingRect(contour)
        #TODO: do I need to keep lnImage
        ln_mask=self.mask[y:y+h,x:x+w]
        ln_image=self.slide[y:y+h,x:x+w]
        new=thresh[y:y+h,x:x+w]
        #we update the contour of the ln based on new mask/image
        contours,_= cv2.findContours(new,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        areas=[cv2.contourArea(c) for c in contours]
        ln_contour = contours[areas.index(max(areas))]

        return LymphNode(ln_contour,self,ln_mask,new,ln_image,'test',germ_label,sinus_label)


class Germinals():
    def __init__(self, ln, mask, label):
        self.ln=ln
        mask[mask!=label]=0
        self.mask=mask
        self.label=label
        self.annMask=mask
        self._germinals=None
        self._num=None
        self._boundingBoxes=None
        self._sizes=None
        self._areas=None

    @property
    def locations(self):
        if self._germinals is None:
            raise ValueError('No germinals detected')
        return [self.centre(g) for g in self._germinals]

    @property
    def total_area(self):
        if self._areas is None:
            raise ValueError('germinal areas not calculated')
        #print('areas', self._areas)
        return sum(self._areas)


    def centre(self,c):
        M = cv2.moments(c)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

        return x, y

    def detect_germinals(self):

        if len(self.mask.shape)==3:
            gray=cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        #edges=cv2.Canny(self.mask,30,200)
        blur=cv2.bilateralFilter(self.mask,9,100,100)
        blur=blur.astype(np.uint8)
        _,thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

        self._germinals=contours
        self._num=len(self._germinals)
        self.ann_mask=thresh
        return self._num


    def measure_sizes(self, pixels=False):

        if len(self._germinals)==0:
            self._sizes=[(0,0)]
        else:
            self._bounding_boxes=list(map(cv2.boundingRect, self._germinals))
            self._sizes=[(b[2],b[3]) for b in self._bounding_boxes]

        if not pixels:
            f = lambda x: (x[0]*self.ln.slide.w_scale,x[1]*self.ln.slide.h_scale)
            self._sizes=list(map(f, self._sizes))
        return self._sizes


    def measure_areas(self, pixels=False):

        self._areas=list(map(cv2.contourArea, self._germinals))
        if not pixels:
            f = lambda x: (x*self.ln.slide.w_scale*self.ln.slide.h_scale)
            self._areas=list(map(f, self._areas))
        return self._areas


    def circularity(self):

        areas=self.measure_areas(pixels=True)
        f = lambda x: cv2.arcLength(x,True)
        perimeters = list(map(f, self._germinals))
        f = lambda x: (4*np.pi*x[0])/np.square(x[1])
        c= list(map(f,zip(areas,perimeters)))
        return c


    def dist_to_centre(self, pixels=False):

        ln_cent_pt=np.asarray(self.ln.centre)
        locations=[np.asarray(l) for l in self.locations]
        f = lambda x: np.linalg.norm(ln_cent_pt-x)
        distances = list(map(f, self.locations))
        return distances


    def dist_to_boundary(self, pixels=False):

        pnts=np.asarray([p for p in self.locations])
        points=[np.asarray([list(p[0]) for p in self.ln.contour])
                           for c in self._germinals]

        f = lambda x: np.sqrt(np.sum((x[0]-x[1])**2,axis=1))
        dist = list(map(f, zip(points,pnts)))
        min_idx = list(map(np.argmin,dist))
        return [d[i] for i,d in zip(min_idx,dist)]


    def visualiseGerminals(self, color=(0,0,255)):

        plot=self.ann_mask
        if self._germinals != 0 and len(self.ann_mask.shape)==2:
            #self.annMask=cv2.cvtColor(self.annMask,cv2.COLOR_GRAY2BGR)
            plot=cv2.drawContours(self.ann_mask, self._germinals, -1, color, 3)

        if self._sizes != [(0,0)]:
            colorReverse=color[::-1]
            for b in self._boundingBoxes:
                x,y,w,h = b
                plot=cv2.rectangle(plot,(x,y),(x+w,y+h), 180,1)

        return plot


class Sinuses():
    def __init__(self, ln, mask, label):
        self.ln=ln
        mask[mask!=label]=0
        self.sinus_mask=mask
        self.ann_mask=mask
        self.label=label
        self._sinuses = None
        self._num=None
        self._areas = None


    @property
    def total_area(self):
        return (len(self.sinus_mask[self.sinus_mask==self.label])
                   *self.ln.slide.w_scale*self.ln.slide.h_scale)


    def detect_sinuses(self):

        if len(self.sinus_mask.shape)==3:
            self.ann_mask=cv2.cvtColor(self.ann_mask, cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(self.ann_mask,30,200)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

        contours = list(filter(lambda x: cv2.contourArea(x) > 0, contours))
        self._sinuses=contours
        self._num=len(self._sinuses)
        self.ann_mask=edges
        return self._num


    def visualise_sinus(self, color=(0,0,255)):
        plot=self.ann_mask
        if self._sinuses != None and len(self.ann_mask.shape)==2:
            self.ann_mask=cv2.cvtColor(self.ann_mask,cv2.COLOR_GRAY2BGR)
            plot=cv2.drawContours(self.ann_mask, self._sinuses, -1, color,1)
        return plot



class LymphNode():
    def __init__(
        self, 
        contour, 
        slide, 
        mask,
        new,
        image,         
        name, 
        germ_label, 
        sinus_label):

        self.slide=slide
        self.contour=contour
        self.mask=mask
        self.new=new
        self.image=image
        #self.area=cv2.contourArea(contour)
        self.germinals = Germinals(self,mask.copy(), germ_label)
        self.sinuses = Sinuses(self,mask.copy(), sinus_label)

    @property
    def area(self):
        a=cv2.contourArea(self.contour)
        return a*self.slide.w_scale*self.slide.h_scale

    @property
    def centre(self):
        M = cv2.moments(self.contour)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

        return x, y

    def visualise(self):

        plot=cv2.cvtColors(self.mask,cv2.COLOR_GRAY2BGR)
        plot=cv2.drawContours(plot, self.node, -1, (0,0,255),1)
        x,y,w,h = self.boundingBox
        plot=cv2.rectangle(plot,(x,y),(x+w,y+h), (255,0,0),1)

        return plot
