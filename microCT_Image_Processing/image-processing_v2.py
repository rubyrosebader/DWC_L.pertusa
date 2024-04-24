#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:49:24 2024

@author: ruby

image processing pipeline of coral skeletons version 2, modified fom marta's script'

From CT images in .tif format:
        1. Density calibration
        2. Filtering.
        3. Segmentation.
        4. Connected components.
        5. Cropping.
        6. Saving cropped images into .mhd format

Input: Path to image data and sample name. Path to output folder

Parameters: 
    sigma: sigma value for gaussian filter
    threshold: threshold value
    calibration curve: slope and intercept of the calibration curve (rho = a+b*I)
    
Output: calibrated and binary cropped images in .mhd format
"""
import argparse
import re 
import os
from glob import glob
from os import path
import SimpleITK as sitk
from skimage.measure import label   
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


input_folder = '/Volumes/T7 Shield/T2_Registered'

coral_pattern = r'^SAMPLE-.*$'

# List all folders in the directory
coral_folders = [folder for folder in os.listdir(input_folder) if re.match(coral_pattern, folder)]

    
def getInputs(coral_folder):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--coral',
                        default=coral_folder,
                        type=str,
                        help='coral skeleton scan to analyse'
                        )
    parser.add_argument('-s', '--sigma',
                        default=0.001,
                        type=float,
                        help='sigma of the gaussian filter'
                        )
    parser.add_argument('-t', '--threshold',
                        default=2.1,
                        type=float,
                        help='sigma of the gaussian filter'
                        )
    parser.add_argument('-a', '--intercept',
                        default=0.918299093,
                        type=float,
                        help='intercept of calibration curve'
                        )
    parser.add_argument('-b', '--slope',
                        default=3.55e-05,
                        type=float,
                        help='slope of calibration curve'
                        )    
    parser.add_argument('-i', '--inp',
                        default='/Volumes/T7 Shield/T2_Registered',
                        type=str,
                        help='input folder'
                        )
    parser.add_argument('-w', '--write',
                        default='/Volumes/T7 Shield/T2_Processed',
                        type=str,
                        help='output folder'
                        )

    args = parser.parse_args()
    coral = args.coral
    inp = args.inp
    sigma = args.sigma
    threshold = args.threshold
    a = args.intercept
    b = args.slope
    write = args.write

    return coral, sigma, threshold, a, b, inp, write

def density_calibration (image, shift, scale):
    '''
    Rescale image intensity (first shifting and then scaling)
    output_pixel = (input_pixel + shift)*scale
    '''
    return sitk.ShiftScale(image, shift, scale)
        
def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest).astype(int)
    return labels_max

def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)

if __name__ == "__main__":
    
    input_folder = '/Volumes/T7 Shield/T2_Registered'
    output_folder = '/Volumes/T7 Shield/T2_Processed'
    
    # Initialize tqdm with the total number of coral_folders
    with tqdm(total=len(coral_folders)) as pbar:
        for coral_folder in coral_folders:
            coral, sigma, threshold, aa, bb, inp, write = getInputs(coral_folder)

            case = os.path.join(input_folder, coral_folder)
            Dir = case
            outPath = path.join(write, coral_folder)
            #print(outPath)
            
            isExist = os.path.exists(outPath)
            
            print("reading scan ", coral)
            print('writing data to', outPath)
            
            if not isExist:
                os.makedirs(outPath)
                print("New output directory is created!")
                    
                files = os.listdir(Dir)
                
                Dir_T0 = '/Volumes/T7 Shield/T0_Registered/'
                try:
                    info = open(path.join(Dir_T0, 'SAMPLE-' + coral[7:11] + 'T0/' , coral[7:11] + 'T0.info'), 'r')
                    print(info)
                except FileNotFoundError:
                    alternative_path = '/Volumes/T7 Shield/T0_Registered/SAMPLE-001-T0/001-T0.info'
                    print("Chose alternative path:", alternative_path)
                    info = open(alternative_path, 'r')
                ps = info.readlines()[2][10:20]
                print(ps)
                
                pattern = os.path.join(Dir, "0*.tif")
                name = glob(pattern)
                #print(name)
                
                Reader = sitk.ImageSeriesReader()
                Reader.SetFileNames(name)
                Image = Reader.Execute()
                Origin = Image.GetOrigin()
                Direction = Image.GetDirection()
                Size = Image.GetSize()
                Spacing = [float(ps),float(ps),float(ps)]
                print(Spacing)
                Image.SetSpacing(Spacing)
                
                print("ps:", ps)
                
                for line in info.readlines():
                    print(line)
                
                Reader = sitk.ImageSeriesReader()
                
                _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
                show_plane(a, sitk.GetArrayFromImage(Image)[Size[2] // 2], title=f'Plane = {Size[2] // 2}')
                show_plane(b, sitk.GetArrayFromImage(Image)[:, Size[1] // 2, :], title=f'Row = {Size[1] // 2}')
                show_plane(c, sitk.GetArrayFromImage(Image)[:, :, Size[0] // 2], title=f'Column = {Size[0] // 2}') 
                print("Image stack and metadata read")   
                
                print ("Filtering image...")  
                gaussfilt = sitk.SmoothingRecursiveGaussianImageFilter()
                gaussfilt.SetSigma(sigma) 
                filteredImage = gaussfilt.Execute(Image)  
                
                _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
                show_plane(a, sitk.GetArrayFromImage(filteredImage)[Size[2] // 2], title=f'Plane = {Size[2] // 2}')
                show_plane(b, sitk.GetArrayFromImage(filteredImage)[:, Size[1] // 2, :], title=f'Row = {Size[1] // 2}')
                show_plane(c, sitk.GetArrayFromImage(filteredImage)[:, :, Size[0] // 2], title=f'Column = {Size[0] // 2}') 
                print ("Image filtered")  
                
                print("Rescaling intensity...")
                imageRescale = density_calibration(filteredImage, aa/bb, bb)
                imageRescale = sitk.Cast(imageRescale,filteredImage.GetPixelIDValue())
                print("Rescaling done")
                    
                print("Thresholding image...")
                threshFilter = sitk.BinaryThresholdImageFilter()
                threshFilter.SetLowerThreshold(threshold)
                threshFilter.SetUpperThreshold(2e16)
                imageBinary = threshFilter.Execute(imageRescale)
                imageBinary.SetDirection(Direction)
                conn = sitk.GetArrayFromImage(sitk.ConnectedComponent(imageBinary))
                largest = getLargestCC(conn)
                del conn, imageBinary
                img_largest = sitk.GetImageFromArray(largest)
                img_largest.SetOrigin(Origin)
                img_largest.SetSpacing(Spacing)
                del largest
                threshFilter.SetLowerThreshold(1.)
                threshFilter.SetUpperThreshold(1.)
                imageBinary = threshFilter.Execute(sitk.ConnectedComponent(img_largest))
                imageBinary.SetDirection(Direction)
                del img_largest
                _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
                show_plane(a, sitk.GetArrayFromImage(imageBinary)[Size[2] // 2], title=f'Plane = {Size[2] // 2}')
                show_plane(b, sitk.GetArrayFromImage(imageBinary)[:, Size[1] // 2, :], title=f'Row = {Size[1] // 2}')
                show_plane(c, sitk.GetArrayFromImage(imageBinary)[:, :, Size[0] // 2], title=f'Column = {Size[0] // 2}') 
                print ("Image thresholded")  
    
                # Creating mask - Filtered image * binary image
                print("Masking image...")
                maskImage = sitk.Multiply(filteredImage, sitk.Cast(imageBinary,filteredImage.GetPixelIDValue()))
                _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
                show_plane(a, sitk.GetArrayFromImage(maskImage)[Size[2] // 2], title=f'Plane = {Size[2] // 2}')
                show_plane(b, sitk.GetArrayFromImage(maskImage)[:, Size[1] // 2, :], title=f'Row = {Size[1] // 2}')
                show_plane(c, sitk.GetArrayFromImage(maskImage)[:, :, Size[0] // 2], title=f'Column = {Size[0] // 2}') 
                print ("Image masked")  
                
                print("Cropping images to bounding box...")
                # Compute statistics
                stats = sitk.LabelStatisticsImageFilter()
                stats.Execute(filteredImage, imageBinary)
                ROI = stats.GetRegion(1)
                cropFilter = sitk.RegionOfInterestImageFilter()
                cropFilter.SetIndex(ROI[0:3])
                cropFilter.SetSize(ROI[3:6])
                croppedImage = cropFilter.Execute(filteredImage)
                croppedBinary = cropFilter.Execute(imageBinary)
                croppedMask = cropFilter.Execute(maskImage)
                croppedSize = croppedImage.GetSize()
                _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
                show_plane(a, sitk.GetArrayFromImage(croppedMask)[croppedSize[2] // 2], title=f'Plane = {croppedSize[2] // 2}')
                show_plane(b, sitk.GetArrayFromImage(croppedMask)[:, croppedSize[1] // 2, :], title=f'Row = {croppedSize[1] // 2}')
                show_plane(c, sitk.GetArrayFromImage(croppedMask)[:, :, croppedSize[0] // 2], title=f'Column = {croppedSize[0] // 2}') 
                np.savetxt(path.join(outPath, coral + "_ROI.txt"), 
                            ROI, fmt="%s")    
                print("Images cropped")
                
                print("Saving cropped images...")
                # Saving cropped images
                sitk.WriteImage(croppedImage, path.join(outPath, coral + "-cropped.mhd"))
                sitk.WriteImage(croppedBinary, path.join(outPath, coral + "-cropped-binary.mhd"))
                print("Images saved")
                
                # Update the progress bar
                pbar.update(1)
    
