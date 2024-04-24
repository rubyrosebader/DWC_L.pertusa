# -*- coding: utf-8 -*-
"""
Adapted on Thur March 28 10:56:00 2024

@author: Ruby Rose Bader
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:39:51 2023

@author: mp2019
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:34:51 2023

@author: mp2019
"""

"""

Measurement of microCT images

Input needed: calibrated and binary image of equal size

From CT images in .mhd format:
        1. Compute volume, surface area, surface area to volume ratio,
            centroid and weithed centroid
        2. Compute inertia tensor (in case this is needed)
        3. Compute local thickness
        4. Save thickness map as image
        
Input: Path to folder containing images and sample name (please check notation)
       Path to output folder

Parameters: no needed

Output: .csv file containing measurements, .txt file with inertia tensor,
        .mhd local thickness image data

"""

#  Version 0.22.0 of scikit-image

import re 
import argparse
from glob import glob
from os import path
import os
import skimage.measure
from skimage import measure
import SimpleITK as sitk
from skimage.measure import label   
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage import img_as_float, img_as_uint
from skimage.measure import regionprops, regionprops_table, marching_cubes, mesh_surface_area
from skimage.morphology import convex_hull_image
from skimage.segmentation import find_boundaries
import pandas as pd
import matplotlib as mpl
from cycler import cycler
import localthickness as lt

input_folder = '/Volumes/T7 Shield/T2_Processed'

coral_pattern = r'^SAMPLE-((029|030|031|032|033|034|035|038|039|041|042|043|044|045|048|049)).*$'

#((028|030))

# List all folders in the directory
coral_folders = [folder for folder in os.listdir(input_folder) if re.match(coral_pattern, folder)]
print(coral_folders)

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])

plt.rc('font', family='Helvetica')

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def getInputs(coral_folder):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--coral',
                        default=coral_folder,
                        type=str,
                        help='coral skeleton scan to analyse'
                        )
    
    parser.add_argument('-i', '--inp',
                        default='/Volumes/T7 Shield/T2_Processed',
                        type=str,
                        help='input folder'
                        )
    parser.add_argument('-w', '--write',
                        default='/Volumes/T7 Shield/T2_Measured',
                        type=str,
                        help='output folder'
                        )    
    parser.add_argument('-th', '--thickness',
                        default='off',
                        type=str,
                        help='output folder'
                        )

    args = parser.parse_args()
    coral = args.coral
    inp = args.inp
    write = args.write
    computeThickness = args.thickness

    return coral, inp, write, strtobool(computeThickness)

def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))
        

def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)

if __name__ == "__main__":
    
    input_folder = '/Volumes/T7 Shield/T2_Processed'
    output_folder = '/Volumes/T7 Shield/T2_Measured'
    
    # Initialize tqdm with the total number of coral_folders
    with tqdm(total=len(coral_folders)) as pbar:
        for coral_folder in coral_folders:
            coral, inp, write, computeThickness = getInputs(coral_folder)
        
            case = glob(path.join(inp, coral))[0]
            print(case)
            Dir = case
            outPath = path.join(write, coral)
            # Check whether the specified path exists or not
            isExist = os.path.exists(outPath)
                
            print("reading scan ", coral)
            print('writing data to', outPath)
            
            
            # Check whether the specified path exists or not
            isExist = os.path.exists(outPath)
            if not isExist:
                # Create a new directory because it does not exist 
                os.makedirs(outPath)
                print("New output directory is created!")
                
                
            #-Read data stack
            print("reading data...")
            
            image = sitk.ReadImage(path.join(inp, coral, coral + '-cropped.mhd'))
            imageBinary = sitk.ReadImage(path.join(inp, coral, coral + '-cropped-binary.mhd'))
            
            # Get bounding box of images
            imageSize = image.GetSize()
            pixelSize = image.GetSpacing()
            
            # Display image
            _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
            show_plane(a, sitk.GetArrayFromImage(image)[imageSize[2] // 2], title=f'Plane = {imageSize[2] // 2}')
            show_plane(b, sitk.GetArrayFromImage(image)[:, imageSize[1] // 2, :], title=f'Row = {imageSize[1] // 2}')
            show_plane(c, sitk.GetArrayFromImage(image)[:, :, imageSize[0] // 2], title=f'Column = {imageSize[0] // 2}')   
            print("Image stack and metadata read")   
            
            print("Computing measurments...")
            props=('label',
                    'area',
                    'bbox',
                    'bbox_area',
                    'centroid',
                    'weighted_centroid',
                    'equivalent_diameter',
                    'extent',
                    'inertia_tensor',
                    'inertia_tensor_eigvals')
            
            propsDist = regionprops_table(sitk.GetArrayFromImage(imageBinary), sitk.GetArrayFromImage(image), properties=props)
            tableProps = pd.DataFrame(propsDist)
             
            bbox = [tableProps['bbox-3'][0]*pixelSize[0],tableProps['bbox-4'][0]*pixelSize[0],tableProps['bbox-5'][0]*pixelSize[0]]
            # Volume
            volume =  tableProps.area[0]*pixelSize[0]**3  
            # Axis length from eigenvalues of the inertia tensor
            s1 = tableProps['inertia_tensor_eigvals-0'][0]
            s2 = tableProps['inertia_tensor_eigvals-1'][0]
            s3 = tableProps['inertia_tensor_eigvals-2'][0]
            
            # Surface area
            sitk.GetArrayFromImage(imageBinary)[find_boundaries(sitk.GetArrayFromImage(imageBinary), mode='outer')] = 0
            # from 19.3 release notes: The deprecated skimage.measure.marching_cubes_lewiner function has been removed (use skimage.measure.marching_cubes instead).
            # Marching cubes algorithm to find surfaces in 3d volumetric data.
            vts, fs, ns, cs = skimage.measure.marching_cubes(sitk.GetArrayFromImage(imageBinary), level=0, spacing=pixelSize, method='lewiner')
            n = len(regionprops(sitk.GetArrayFromImage(imageBinary)))
            lst = [[] for j in range(n+1)]
            for j in fs: lst[int(cs[j[0]])].append(j)
            areas = [0 if len(j)==0 else mesh_surface_area(vts, j) for j in lst]
            area = np.asarray(areas)
            area = area[1:]
               
            columns = ['voxel_size_x in mm',
                       'voxel_size_y in mm',
                       'voxel_size_z in mm',
                    'area in mm2',
                    'volume in mm3',
                    'SA_Vol in mm-1',
                    'centroid_x in pixels',
                    'centroid_y in pixels',
                    'centroid_z in pixels',
                    'weighted_centroid_x in pixels',
                    'weighted_centroid_y in pixels',
                    'weighted_centroid_z in pixels']
            
            table = np.c_[pixelSize[0],pixelSize[1],pixelSize[2],
                     area,
                     volume,
                     area/volume,
                     tableProps['centroid-2'][0],
                     tableProps['centroid-1'][0],
                     tableProps['centroid-0'][0],
                     tableProps['weighted_centroid-2'][0],
                     tableProps['weighted_centroid-1'][0],
                     tableProps['weighted_centroid-0'][0]]
            
            df = pd.DataFrame(data=table, columns=columns)
            df = df.transpose()
            df.to_csv(path.join(outPath, coral + "-morphometry.csv"))
            
            InertiaTensor = np.asarray([[tableProps['inertia_tensor-0-0'][0],tableProps['inertia_tensor-0-1'][0],tableProps['inertia_tensor-0-2'][0]],
                              [tableProps['inertia_tensor-1-0'][0],tableProps['inertia_tensor-1-1'][0],tableProps['inertia_tensor-1-2'][0]],
                              [tableProps['inertia_tensor-2-0'][0],tableProps['inertia_tensor-2-1'][0],tableProps['inertia_tensor-2-2'][0]]])
            np.savetxt(path.join(outPath, coral + "-inertiaTensor.txt"), 
                        InertiaTensor, fmt="%s")
            print("Image measured")
            """
            # Compute 3D thickness
            print("Computing local thickness...")
            thickness_map = lt.local_thickness(sitk.GetArrayFromImage(imageBinary))
            th = np.ndarray.flatten(2*thickness_map*pixelSize[0]*1000)
            th = th[th!=0]
            th_stats = ['mean','median','std','max','min']
            th_stats_data =[np.mean(th), np.median(th), np.std (th), np.min(th), np.max(th)]
            data_dict = {'Statistic': th_stats, 'Value': th_stats_data}
            # Create DataFrame from the dictionary
            df = pd.DataFrame(data=data_dict)
            # Save the DataFrame to CSV
            df.to_csv(path.join(outPath, coral + "-thickness-stats.csv"), index=False)

            #Plot Histogram    
            plt.figure(figsize=(8,8))
            y,x,_=plt.hist(th, 10, alpha=.5, edgecolor='k')
            plt.title("")
            plt.xlabel(r'Thickness in $\mu m$')
            plt.ylabel("Frequency")
            plt.savefig(path.join(outPath, coral + 'thickness-distribution.png'), dpi=300)
            plt.show()
                
                # Display image
            _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
            show_plane(a, thickness_map[imageSize[2] // 2], title=f'Plane = {imageSize[2] // 2}', cmap='jet')
            show_plane(b, thickness_map[:, imageSize[1] // 2, :], title=f'Row = {imageSize[1] // 2}', cmap='jet')
            show_plane(c, thickness_map[:, :, imageSize[0] // 2], title=f'Column = {imageSize[0] // 2}', cmap='jet')
            
            # Save thickness map
            thicknessMap = sitk.GetImageFromArray(2*thickness_map*pixelSize[0])
            thicknessMap.SetSpacing(pixelSize)
            thicknessMap = sitk.Cast(thicknessMap,image.GetPixelIDValue())
            sitk.WriteImage(thicknessMap, path.join(outPath, coral + "-thickness-map-in-mm.mhd"))
            print("Local thickness...")
        
            # Update the progress bar
            pbar.update(1)
                
                """