
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Canvas():
    def __init__(self,x_dim,y_dim):
        canvas=np.zeros((int(x_dim+1),int(y_dim+1)))
        self.canvas=canvas
    
    #@property
    #def canvas(self):
        #return self._canvas
       
    #@canvas.setter
    def stitch(self,m,x,y):
        self.canvas[x:x+m.shape[0],y:y+m.shape[1]]=m


def stitch(canvas, mask, x, y, h, w, t_dim, step, margin):
    #Top left
    if (y==0) and (x==0):
        m=mask[0:margin+step,0:margin+step]
        canvas.stitch(m,x,y)
    #Top right
    elif (y==h-t_dim) and (x==0):
        m=mask[0:margin+step,margin:t_dim]
        canvas.stitch(m,x,y+margin)
    #lower left
    elif (y==0 and x==w-t_dim):
        m=mask[margin:t_dim,0:margin+step]
        canvas.stitch(m,x+margin,y)
    #lower right
    elif (y==h-t_dim) and (x==w-t_dim):
        m=mask[margin:t_dim,margin:t_dim]
        canvas.stitch(m,x+margin,y+margin)
    #left
    elif y==0:
        m=mask[margin:margin+step,0:margin+step]
        canvas.stitch(m,x+margin,y)
    #right
    elif y==h-t_dim:
        m=mask[margin:margin+step,margin:t_dim]
        #print(canvas.stitch)
        canvas.stitch(m,x+margin,y+margin)
    #top
    elif x==0:
        m=mask[0:margin+step,margin:margin+step]
        canvas.stitch(m,x,y+margin)
    #bottom
    elif x==w-t_dim:
        m=mask[margin:t_dim,margin:margin+step]
        canvas.stitch(m,x+margin,y+margin)
    #middle
    else:
        m=mask[margin:margin+step,margin:margin+step]
        canvas.stitch(m,x+margin,y+margin)
    return canvas
