import tifffile as tiff
import numpy as np
import skimage
import os
import scipy
import time
#import pandas as pd

#parameters
wells=[0]
n_embryos=[1]
n_time=97
destination3d = "/rds/user/ay343/hpc-work/validation/data_3d/"
destination2d = "/rds/user/ay343/hpc-work/validation/data_2d/"

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
    well_source = f"/rds/project/rds-1FbiQayZlSY/cmp/original_embryo/"
    for embryo_idx in range(n_embryos[well_idx]):
        em_source = well_source+f"embryo_{embryo_idx+1}/"
        shape_old=list(tiff.imread(em_source+f"intensities/t0001_DNA.tif").shape)
        shape_new = [shape_old[0], shape_old[1]//2, shape_old[2]//2]
        #produce resized versions, in 3D and 2D
        for t in range(n_time):
            t_str=str(t+1).zfill(4)
            elapsed_time = time.time() - start_time
            print(f"well{well}, em{embryo_idx+1}, t{t_str}: {elapsed_time:.2f} seconds elapsed")

            ##process anisotropic
            img=tiff.imread(em_source+f"intensities/t{t_str}_DNA.tif")
            img = img[:,0:shape_new[1]*2,0:shape_new[2]*2]
            img=(img[:,0::2,0::2]+img[:,0::2,1::2]+
                 img[:,1::2,0::2]+img[:,1::2,1::2])/4
            tiff.imwrite(destination3d+f"anisotropic/t{t_str}_img.tif", img)

            lbl=tiff.imread(em_source+f"labels/t{t_str}_DNA_label.tif")
            lbl = lbl[:,0:shape_new[1]*2,0:shape_new[2]*2]
            lbl = skimage.measure.block_reduce(lbl, (1,2,2), np.max).astype(np.uint8)
            tiff.imwrite(destination3d+f"anisotropic/t{t_str}_lbl.tif", lbl)

            div_lbl=tiff.imread(em_source+f"division_labels/t{t_str}_division_labels.tif")
            div_lbl = div_lbl[:,0:shape_new[1]*2,0:shape_new[2]*2]
            div_lbl = skimage.measure.block_reduce(div_lbl, (1,2,2), np.max).astype(np.uint8)
            tiff.imwrite(destination3d+f"anisotropic/t{t_str}_div.tif", div_lbl)

            get_slices(img,lbl,div_lbl,destination2d+f"both/t{t_str}",1,axis=0)
            get_slices(img,lbl,None,destination2d+f"anisotropic/t{t_str}",3,axis=1)
            get_slices(img,lbl,None,destination2d+f"anisotropic/t{t_str}",3,axis=2)
            
            ##process isotropic
            img=scipy.ndimage.zoom(img,[1/0.174,1,1],order=1,cval=99)
            tiff.imwrite(destination3d+f"isotropic/t{t_str}_img.tif", img)
            
            lbl=scipy.ndimage.zoom(lbl,[1/0.174,1,1],order=0,cval=0)
            tiff.imwrite(destination3d+f"isotropic/t{t_str}_lbl.tif", lbl)

            div_lbl=scipy.ndimage.zoom(div_lbl,[1/0.174,1,1],order=0,cval=0)
            tiff.imwrite(destination3d+f"isotropic/t{t_str}_div.tif", div_lbl)
            
            get_slices(img,lbl,None,destination2d+f"isotropic/t{t_str}",12,axis=1)
            get_slices(img,lbl,None,destination2d+f"isotropic/t{t_str}",12,axis=2)