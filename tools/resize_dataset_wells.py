import tifffile as tiff
import numpy as np
import skimage
import os
import scipy
import time
#import pandas as pd

#parameters
wells=[3,5,6]
n_embryos=[3,4,4]
n_time=120
destination3d = "/rds/user/ay343/hpc-work/data_3d/"
destination2d = "/rds/user/ay343/hpc-work/data_2d/"

#define things to use later
start_time = time.time()
np.random.seed(1729)

def get_slices(img, lbl, div, destination, sampling_rate=1, axis=0, min_cells=3):
    #given 3d block, save 2d slices
    if axis==1:
        img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
        lbl = np.moveaxis(lbl, [0, 1, 2], [1, 2, 0])
        if div is not None:
            div = np.moveaxis(div, [0, 1, 2], [1, 2, 0])
    elif axis==2:
        img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
        lbl = np.moveaxis(lbl, [0, 1, 2], [2, 1, 0])
        if div is not None:
            div = np.moveaxis(div, [0, 1, 2], [2, 1, 0])
    for sliceno in range(np.random.randint(sampling_rate),len(lbl),sampling_rate):
        slice_lbl=lbl[sliceno]
        if len(np.unique(slice_lbl))<=min_cells: #ignore slices with less than three cells
            continue
        slice=img[sliceno]
        sliceno_str = str(sliceno+1).zfill(4)
        tiff.imwrite(destination+f"_a{axis}_s{sliceno_str}_img.tif",slice)
        tiff.imwrite(destination+f"_a{axis}_s{sliceno_str}_lbl.tif",slice_lbl)
        if div is not None:
            slice_div=div[sliceno]
            if len(np.unique(slice_div))>1:
                tiff.imwrite(destination+f"_a{axis}_s{sliceno_str}_div.tif",slice_div)
    return

for well_idx,well in enumerate(wells):
    #specify where the data is
    well_source = f"/rds/project/rds-1FbiQayZlSY/cmp/well{well}_DNA/"
    for embryo_idx in range(n_embryos[well_idx]):
        #if well_idx!=2 | embryo_idx!=3:
        #    continue
        em_source = well_source+f"embryo_{embryo_idx+1}/"
        crop_params = np.load(em_source+"crop.npy")
        shape_old = [crop_params[0,1]-crop_params[0,0],crop_params[1,1]-crop_params[1,0],crop_params[2,1]-crop_params[2,0]]
        shape_new = [shape_old[0], shape_old[1]//2, shape_old[2]//2]
        #produce resized versions, in 3D and 2D
        for t in range(n_time):
            t_str=str(t+1).zfill(4)
            elapsed_time = time.time() - start_time
            print(f"well{well}, em{embryo_idx+1}, t{t_str}: {elapsed_time:.2f} seconds elapsed")

            ##process anisotropic
            img=tiff.imread(em_source+f"masked_intensities/t{t_str}_img.tif")
            img = img[:,0:shape_new[1]*2,0:shape_new[2]*2]
            img=(img[:,0::2,0::2]+img[:,0::2,1::2]+
                 img[:,1::2,0::2]+img[:,1::2,1::2])/4
            tiff.imwrite(destination3d+f"anisotropic/intensities/well{well}_em{embryo_idx+1}_t{t_str}_img.tif", img)

            lbl=tiff.imread(em_source+f"track_labels/t{t_str}_mcherry_label_track.tif")
            lbl = lbl[:,0:shape_new[1]*2,0:shape_new[2]*2]
            lbl = skimage.measure.block_reduce(lbl, (1,2,2), np.max).astype(np.uint8)
            tiff.imwrite(destination3d+f"anisotropic/labels/well{well}_em{embryo_idx+1}_t{t_str}_lbl.tif", lbl)

            div_lbl=tiff.imread(em_source+f"division_labels/t{t_str}_div_lbl.tif")
            div_lbl = div_lbl[:,0:shape_new[1]*2,0:shape_new[2]*2]
            div_lbl = skimage.measure.block_reduce(div_lbl, (1,2,2), np.max).astype(np.uint8)
            tiff.imwrite(destination3d+f"anisotropic/div_labels/well{well}_em{embryo_idx+1}_t{t_str}_div.tif", div_lbl)

            get_slices(img,lbl,div_lbl,destination2d+f"both/well{well}_em{embryo_idx+1}_t{t_str}",1,axis=0)
            get_slices(img,lbl,None,destination2d+f"anisotropic/well{well}_em{embryo_idx+1}_t{t_str}",3,axis=1)
            get_slices(img,lbl,None,destination2d+f"anisotropic/well{well}_em{embryo_idx+1}_t{t_str}",3,axis=2)
            
            ##process isotropic
            img=scipy.ndimage.zoom(img,[1/0.174,1,1],order=1,cval=99)
            tiff.imwrite(destination3d+f"isotropic/intensities/well{well}_em{embryo_idx+1}_t{t_str}_img.tif", img)
            
            lbl=scipy.ndimage.zoom(lbl,[1/0.174,1,1],order=0,cval=0)
            tiff.imwrite(destination3d+f"isotropic/labels/well{well}_em{embryo_idx+1}_t{t_str}_lbl.tif", lbl)

            div_lbl=scipy.ndimage.zoom(div_lbl,[1/0.174,1,1],order=0,cval=0)
            tiff.imwrite(destination3d+f"isotropic/div_labels/well{well}_em{embryo_idx+1}_t{t_str}_div.tif", div_lbl)

            get_slices(img,lbl,None,destination2d+f"isotropic/well{well}_em{embryo_idx+1}_t{t_str}",12,axis=1)
            get_slices(img,lbl,None,destination2d+f"isotropic/well{well}_em{embryo_idx+1}_t{t_str}",12,axis=2)