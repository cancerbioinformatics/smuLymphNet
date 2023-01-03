import os
import glob
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import openslide
import pandas as pd

import measure as me
#from src.utilities.utils import getFiles

def getFiles(filesPath, ext):
    filesLst=[]
    for path, subdirs, files in os.walk(filesPath):
        for name in files:
            if name.endswith(ext):
                filesLst.append(os.path.join(path,name))
    return filesLst


def ln_quantification(mask,slide)

    dims=slide.dimensions
    slide_thumb=np.array(wsi.get_thumbnail(size=wsi.level_dimensions[6]))

    #mdims=mask.shape
    slide_thumb=np.array(wsi.get_thumbnail(size=wsi.level_dimensions[6]))
    #mx,my=wsi.level_dimensions[6]
    mask=cv2.resize(mask,wsi.level_dimensions[6])
    mdims=image.shape
    mask[:,:,0][mask[:,:,0]==128]=0
    mask[:,:,1][mask[:,:,1]==255]=0
    mask[:,:,2][mask[:,:,2]==128]=0
    mask[:,:,2][mask[:,:,2]==255]=0 
    mask[:,:,0][mask[:,:,0]!=0]=255
    mask[:,:,0][mask[:,:,1]!=0]=128
    mask=mask[:,:,0]

    mShape=mask.shape
    iShape=image.shape
    w=dims[0]
    h=dims[1]
    wNew=mShape[0]
    hNew=mShape[1]
    slide = me.Slide(image,mask,w,h,wNew,hNew)
    num = slide.extractLymphNodes(255,128)
    print('number of ln: {}'.format(num))

    for i, ln in enumerate(slide._lymphNodes):

        mask=cv2.drawContours(ln.image,ln.contour,-1,(0,0,255),3)
        lnAreas.append(ln.area*1e6)
        cv2.imwrite(os.path.join(savePath,name+str(i)+'_ln.png'),mask)
        numGerms=ln.germinals.detectGerminals()
        numSinuses=ln.sinuses.detectSinuses()
        ln.germinals.measureSizes()
        ln.germinals.measureAreas()
        plotS=ln.sinuses.visualiseSinus()
        plotG=ln.germinals.visualiseGerminals()

        sinus_mask=ln.sinuses.sinusMask
        germinal_mask=ln.germinals.mask
        binary_mask=np.zeros((mask.shape))
        binary_mask=cv2.fillPoly(binary_mask,pts=[ln.contour],color=(255,255,255))

        germinal_mask = germinal_mask[:,:,None]*np.ones(3, dtype=int)[None,None,:]
        sinus_mask = sinus_mask[:,:,None]*np.ones(3, dtype=int)[None,None,:]

        binary_mask[germinal_mask==255]=0
        germinal_mask[:,:,0]=0
        germinal_mask[:,:,2]=0

        sinus_mask[sinus_mask==128]=255
        binary_mask[sinus_mask==255]=0
        sinus_mask[:,:,0]=0
        sinus_mask[:,:,1]=0

        binary_mask[:,:,1]=0
        binary_mask[:,:,2]=0

        binary_mask=binary_mask+germinal_mask+sinus_mask
        cv2.imwrite(os.path.join(savePath,name+str(i)+'_binarymask.png'),binary_mask)










def main(wsi_paths, mask_paths, save_path):
    
    names=[]
    lnIdx=[]
    lnAreas=[]
    germNum=[]
    avgGermSizes=[]
    avgGermAreas=[]
    germTotalAreas2=[]
    germTotalAreas=[]
    sinusNum=[]
    totalSinusArea=[]
    avgGermW=[]
    avgGermH=[]
    centDist=[]
    boundDist=[]
    maxGerm=[]
    minGerm=[]
    shapes=[]
    statusLst=[]
    patients=[]
    ln_statuses=[]

    for m_path,s_path in zip(mask_paths,wsi_paths):
        name=os.path.basename(m_path)[:-4]
        patient=maskF.split('/')[-2]
        print('image name: {}'.format(name))
        print('patient name:{}'.format(patient))
        
        mask = cv2.imread(m_path)
        wsi=openslide.OpenSlide(wsiF)
        dims=wsi.dimensions

        #mdims=mask.shape
        slide_thumb=np.array(wsi.get_thumbnail(size=wsi.level_dimensions[6]))
        #mx,my=wsi.level_dimensions[6]
        mask=cv2.resize(mask,wsi.level_dimensions[6])
        mdims=image.shape
        mask[:,:,0][mask[:,:,0]==128]=0
        mask[:,:,1][mask[:,:,1]==255]=0
        mask[:,:,2][mask[:,:,2]==128]=0
        mask[:,:,2][mask[:,:,2]==255]=0 
        mask[:,:,0][mask[:,:,0]!=0]=255
        mask[:,:,0][mask[:,:,1]!=0]=128
        mask=mask[:,:,0]

        mShape=mask.shape
        iShape=image.shape
        w=dims[0]
        h=dims[1]
        wNew=mShape[0]
        hNew=mShape[1]
        slide = me.Slide(image,mask,w,h,wNew,hNew)
        num = slide.extractLymphNodes(255,128)
        print('number of ln: {}'.format(num))

        for i, ln in enumerate(slide._lymphNodes):

            mask=cv2.drawContours(ln.image,ln.contour,-1,(0,0,255),3)
            lnAreas.append(ln.area*1e6)
            cv2.imwrite(os.path.join(savePath,name+str(i)+'_ln.png'),mask)
            numGerms=ln.germinals.detectGerminals()
            numSinuses=ln.sinuses.detectSinuses()
            ln.germinals.measureSizes()
            ln.germinals.measureAreas()
            plotS=ln.sinuses.visualiseSinus()
            plotG=ln.germinals.visualiseGerminals()

            sinus_mask=ln.sinuses.sinusMask
            germinal_mask=ln.germinals.mask
            binary_mask=np.zeros((mask.shape))
            binary_mask=cv2.fillPoly(binary_mask,pts=[ln.contour],color=(255,255,255))

            germinal_mask = germinal_mask[:,:,None]*np.ones(3, dtype=int)[None,None,:]
            sinus_mask = sinus_mask[:,:,None]*np.ones(3, dtype=int)[None,None,:]

            binary_mask[germinal_mask==255]=0
            germinal_mask[:,:,0]=0
            germinal_mask[:,:,2]=0

            sinus_mask[sinus_mask==128]=255
            binary_mask[sinus_mask==255]=0
            sinus_mask[:,:,0]=0
            sinus_mask[:,:,1]=0

            binary_mask[:,:,1]=0
            binary_mask[:,:,2]=0

            print('b',np.unique(binary_mask[:,:,1]))
            print('g',np.unique(binary_mask[:,:,2]))
            binary_mask=binary_mask+germinal_mask+sinus_mask
            print(np.unique(binary_mask))
            cv2.imwrite(os.path.join(savePath,name+str(i)+'_binarymask.png'),binary_mask)
            #f,ax = plt.subplots(1,3,figsize=(15,25))
            #ax[0].imshow(ln.mask, cmap='gray')
            #ax[0].axis('off')
            #ax[1].imshow(plotG, cmap='gray')
            #ax[1].axis('off')
            #ax[2].imshow(plotS, cmap='gray')
            #ax[2].axis('off')
            #plt.show()

            #cv2.imwrite(os.path.join(plotPath,name+str(i)+'_sinus.png'),plotS)
            #cv2.imwrite(os.path.join(plotPath,name+str(i)+'_germs.png'),plotG)

            sizes=ln.germinals._sizes
            if sizes==[(0,0)]:
                avgSizes=[0,0]
            else:
                avgSizes=np.mean(list(zip(*sizes)),axis=1)
            areas=ln.germinals._areas
            if len(areas)==0:
                areas=[0]
            else:
                areas

            avgGermArea2=np.mean(areas)
            maxGermArea2=np.max(areas)
            minGermArea2=np.min(areas)
            germArea=ln.germinals.totalArea
            germArea2=ln.germinals.totalArea2
            sinusArea=ln.sinuses.totalArea2
            germDistCent=ln.germinals.distanceFromCenter()
            germDistBoundary=ln.germinals.distanceFromBoundary()
            germShape=np.mean(ln.germinals.circularity())
            names.append(name)
            lnIdx.append(i)
            statusLst.append(status)
            germNum.append(numGerms)
            avgGermW.append(np.round(avgSizes[0]*1e6,2))
            avgGermH.append(np.round(avgSizes[1]*1e6,2))
            germTotalAreas.append(np.round(germArea*1e6,2))
            germTotalAreas2.append(np.round(germArea2*1e6,2))
            avgGermAreas.append(np.round(avgGermArea2*1e6,4))
            maxGerm.append(np.round(maxGermArea2*1e6,4))
            minGerm.append(np.round(minGermArea2*1e6,4))
            shapes.append(germShape)
            sinusNum.append(numSinuses)
            totalSinusArea.append(np.round(sinusArea*1e6,2))
            centDist.append(np.round(np.mean(germDistCent)))
            boundDist.append(np.round(np.mean(germDistBoundary)))
            patients.append(patient)
            ln_statuses.append(ln_status)

    stats={
        'patient':patients,
        'name':names,
        'ln_status':ln_statuses,
        'ln_idx':lnIdx,
        'ln_area':lnAreas,
        'germ_number':germNum,
        'avg_germ_width':avgGermW,
        'avg_germ_height':avgGermH,
        'total_germ_area':germTotalAreas,
        'total_germ_area2':germTotalAreas2,
        'section_status':statusLst,
        'avg_germ_area': avgGermAreas,
        'avg_germ_shape':shapes,
        'max_germ_area': maxGerm,
        'min_germ_area': minGerm,
        'germ_distance_to_centre':centDist,
        'germ_distance_to_boundary':boundDist,
        'sinus_number': sinusNum,
        'total_sinus_area':totalSinusArea

    }

    for k,v in stats.items():
        print(k, len(v))
        statsDf=pd.DataFrame(stats)
    statsDf.to_csv('/home/verghese/node_details_cancer_90552.csv')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument(
        '-wp',
        '--wsipath',
        required=True,
        help='path to wholeslide images'
    )
    ap.add_argument(
        '-mp',
        '--maskpath',
        required=True,
        help='path to prediction masks'
    )
    ap.add_argument(
        '-sp',
        '--savepath',
        required=True,
        help='path to save plots and stats'
    )

    args=vars(ap.parse_args())

    if not batch:
        mask = cv2.imread(args['maskpath'])
        wsi=openslide.OpenSlide(args['wsipath'])

        name=os.path.basename(m_path)[:-4]
        print('image name: {}'.format(name))
        








    slide_paths=getFiles(wsiPath,'ndpi')
    print(f'Slides:n={len(slide_paths)}'
    mask_paths=getFiles(maskPath,'png')

    totalMasks=[t for t in totalMasks if 'image' not in t]
    print(f'Masks:n={len(mask_paths)}'
    
    #all_ln_status=pd.read_csv('/home/verghese/ln_status_3.csv',index_col=['image_name'])

    print('analysing lymph nodes...',flush=True)
    main(slide_paths, mask_paths)
