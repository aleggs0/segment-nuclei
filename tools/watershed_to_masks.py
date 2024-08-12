import tifffile as tiff
import numpy as np
import skimage
import os
import scipy
import time
import pandas as pd

#parameters
wells=[3,5,6]
n_embryos=[3,4,4]
n_time=135
watershed_folder = "/rds/user/ay343/hpc-work/embryo_seg/outputs/"
background_val = 99

#define things to use later
start_time = time.time()
full_shape=[n_time]+list(np.shape(tiff.imread("/rds/project/rds-1FbiQayZlSY/cmp/well3_DNA/intensity/t0001_mcherry.tif")))

def flood_fill_hull(image):    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img


for well_idx,well in enumerate(wells):
    elapsed_time = time.time() - start_time
    print(f"well{well}: {elapsed_time:.2f} seconds elapsed")
    #specify where the data is
    well_source = f"/rds/project/rds-1FbiQayZlSY/cmp/well{well}_DNA/"
    watershed_out_4d = tiff.imread(f"/rds/user/ay343/hpc-work/embryo_seg/well{well}_micro_watershed_4d.tif")
    for embryo_idx in range(n_embryos[well_idx]):
        #new new folders for processed data
        if not os.path.exists(well_source+f"embryo_{embryo_idx+1}/masks/"):
            os.mkdir(well_source+f"embryo_{embryo_idx+1}/masks/")
        if not os.path.exists(well_source+f"embryo_{embryo_idx+1}/masked_intensities/"):
            os.mkdir(well_source+f"embryo_{embryo_idx+1}/masked_intensities/")
        if not os.path.exists(well_source+f"embryo_{embryo_idx+1}/division_labels/"):
            os.mkdir(well_source+f"embryo_{embryo_idx+1}/division_labels/")
        minimask=(watershed_out_4d==embryo_idx+1)
        crop_params=np.load(well_source+f"embryo_{embryo_idx+1}/crop.npy")
        #mask=np.zeros(full_shape,np.bool)
        for t in range(n_time):
            t_str=str(t+1).zfill(4)
            elapsed_time = time.time() - start_time
            print(f"well{well}, em{embryo_idx+1}, time {t}: {elapsed_time:.2f} seconds elapsed")
#get masks
            minimask[t]=flood_fill_hull(minimask[t])
            minimask[t]=scipy.ndimage.binary_dilation(minimask[t], iterations=2)
            mask=minimask[t].repeat(3, axis=0).repeat(36, axis=1).repeat(36, axis=2)[0:full_shape[1],0:full_shape[2],0:full_shape[3]]
            mask = mask[crop_params[0,0]:crop_params[0,1],crop_params[1,0]:crop_params[1,1],crop_params[2,0]:crop_params[2,1]]
            tiff.imwrite(well_source+f"embryo_{embryo_idx+1}/masks/t{t_str}_mask.tif",mask)
#get cropped images
            img=tiff.imread(well_source+f"intensity/t{t_str}_mcherry.tif")
            img=img[crop_params[0,0]:crop_params[0,1],crop_params[1,0]:crop_params[1,1],crop_params[2,0]:crop_params[2,1]]
            img[np.logical_not(mask)]=background_val
            #save the masked, cropped intensities
            tiff.imwrite(well_source+f"embryo_{embryo_idx+1}/masked_intensities/t{t_str}_img.tif",img)
            #initialise a corresponding set of labels for divisions
            tiff.imwrite(well_source+f"embryo_{embryo_idx+1}/division_labels/t{t_str}_div_lbl.tif",np.zeros(mask.shape,np.uint8))

#update the empty div_lbl files with the labels for cell divisions
for well_idx,well in enumerate(wells):
    #specify where the data is
    well_source = f"/rds/project/rds-1FbiQayZlSY/cmp/well{well}_DNA/"
    for embryo_idx in range(n_embryos[well_idx]):
        em_source = well_source+f"embryo_{embryo_idx+1}/"
        crop_params=np.load(em_source+"crop.npy")
        split_df = pd.read_csv(em_source+"split_df.csv")
        tp=split_df.columns.get_loc('timepoint')
        pa=split_df.columns.get_loc('parent')
        pa=split_df.columns.get_loc('parent')
        #track_df = pd.read_csv(em_source+"track_df.csv")
        #parent_df = track_df.merge(split_df,left_on='track_id',right_on='parent',suffixes=('_curr','_split'))
        #parent_df['time_to_split'] = parent_df['timepoint_split']-parent_df['timepoint_curr']
        for idx in range(len(split_df)):
            elapsed_time = time.time() - start_time
            print(f"well{well}, em{embryo_idx+1}, instance {idx}: {elapsed_time:.2f} seconds elapsed")
            for dt in [1,2,3,4,5]:
            #add the labelled cell to the corresponding division labels file
                t = split_df.iloc[idx,tp]-dt
                if t>0:
                    t_str = str(t+1).zfill(4) #str(instance['timepoint_curr']+1).zfill(4)
                    div_lbl = tiff.imread(well_source+f"embryo_{embryo_idx+1}/division_labels/t{t_str}_div_lbl.tif")
                    cell_lbl = tiff.imread(well_source+f"embryo_{embryo_idx+1}/track_labels/t{t_str}_mcherry_label_track.tif")
                    div_lbl[cell_lbl==split_df.iloc[idx,pa]]=6-dt
                    tiff.imwrite(well_source+f"embryo_{embryo_idx+1}/division_labels/t{t_str}_div_lbl.tif",div_lbl)