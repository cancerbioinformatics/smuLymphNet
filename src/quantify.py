import os
import glob
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import openslide
import pandas as pd

import measure as me

node_stats={
        'name': [],
        'ln_idx': [],
        'ln_area': [],
        'germ_number': [],
        'avg_germ_width': [],
        'avg_germ_height': [],
        'total_germ_area': [],
        'avg_germ_area': [],
        'avg_germ_shape': [],
        'max_germ_area': [],
        'min_germ_area': [],
        'germ_distance_to_centre': [],
        'germ_distance_to_boundary': [],
        'total_sinus_area': []
    }


def label_nodes(slide_contour):
    M=cv2.moments(slide_contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return (cx,cy)


def ln_quantification(mask,wsi):

    dims=wsi.dimensions
    x_res=wsi.properties[openslide.PROPERTY_NAME_MPP_X]
    y_res=wsi.properties[openslide.PROPERTY_NAME_MPP_Y]
    
    print(wsi.level_dimensions[0])
    #scale=wsi.level_dimensions[0]/wsi.level_dimensions[1]
    maxres = float(wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    wsi_thumb=np.array(wsi.get_thumbnail(size=wsi.level_dimensions[6]))
    mask=cv2.resize(mask,wsi.level_dimensions[6])
     
    mask[:,:,0][mask[:,:,0]==128]=0
    mask[:,:,1][mask[:,:,1]==255]=0
    mask[:,:,2][mask[:,:,2]==128]=0
    mask[:,:,2][mask[:,:,2]==255]=0        
    #mask=cv2.resize(mask,(mdims[1],mdims[0]))
    mask[:,:,0][mask[:,:,0]!=0]=255
    mask[:,:,0][mask[:,:,1]!=0]=128
    mask=mask[:,:,0]

    slide = me.Slide(wsi_thumb,mask)
    num = slide.locate_nodes(255,128)
    print('number of ln: {}'.format(num))
     
    for i, ln in enumerate(slide._lymphnodes):
        centre=label_nodes(slide.contours[i])
        wsi_thumb=cv2.putText(wsi_thumb, 'LN: ' + str(i),centre,cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),7)

        num_sinuses = ln.sinuses.detect_sinuses()
        num_gcs = ln.germinals.detect_germinals()
        print(f'GCs detected: {num_gcs}')
        
        node_stats['ln_area'].append(ln.area*1e6)
        node_stats['germ_number'].append(num_gcs)
        num_sinuses=ln.sinuses.detect_sinuses()
        ln.germinals.measure_sizes()
        ln.germinals.measure_areas()
        
        print(ln.germinals._sizes[0])
        node_stats['avg_germ_width'].append(np.round(ln.germinals._sizes[0][0]*1e6,2)) 
        node_stats['avg_germ_height'].append(np.round(ln.germinals._sizes[0][1]*1e6,2))
        
        #print('greg',ln.germinals._germinals)
        areas=ln.germinals._areas
        areas=[0] if len(areas)==0 else areas

        node_stats['avg_germ_area'].append(np.round(np.mean(areas)*1e6,4))
        node_stats['min_germ_area'].append(np.round(np.min(areas)*1e6,4))
        node_stats['max_germ_area'].append(np.round(np.max(areas)*1e6,4))
        node_stats['total_germ_area'].append(np.round(np.sum(areas)*1e6,4))
        node_stats['avg_germ_shape'].append(np.mean(ln.germinals.circularity()))
        node_stats['germ_distance_to_centre'].append(ln.germinals.dist_to_centre())
        node_stats['germ_distance_to_boundary'].append(np.mean(ln.germinals.dist_to_boundary()))
        node_stats['total_sinus_area'].append(np.round(ln.sinuses.total_area*1e6,2))
        node_stats['name'].append(name)
        node_stats['ln_idx'].append(i)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument(
        '-wp',
        '--wsi_paths',
        required=True,
        help='path to wholeslide images'
    )
    ap.add_argument(
        '-mp',
        '--mask_paths',
        required=True,
        help='path to prediction masks'
    )
    ap.add_argument(
        '-sp',
        '--save_paths',
        required=True,
        help='path to save plots and stats'
    )

    args=vars(ap.parse_args())
    print('analysing lymph nodes...',flush=True)
    mask_paths=glob.glob(os.path.join(args['mask_paths'],'*'))
    wsi_paths=glob.glob(os.path.join(args['wsi_paths'],'*'))
   
    print(wsi_paths)
    names=[os.path.basename(m) for m in mask_paths]
    mask_paths=[m for _, m in sorted(zip(names,mask_paths))]
    wsi_paths=[w for _, w in sorted(zip(names,wsi_paths))]
    
    for mask_path, wsi_path in zip(mask_paths,wsi_paths):
        mask = cv2.imread(mask_path)
        print(mask_path)
        print(wsi_path)
        wsi = openslide.OpenSlide(wsi_path)

        name=os.path.basename(mask_path)[:-4]
        print('image name: {}'.format(name))
        if '18001019' not in name:
            continue
        ln_quantification(mask,wsi)
        
    for k,v in node_stats.items():
        print(len(v))

    stats_df=pd.DataFrame(node_stats)
    stats_df.to_csv(os.path.join(args['save_paths'],'quantification.csv'))
                                     
