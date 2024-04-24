# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:36:56 2023

@author: mp2019
"""

"""

Phantom calibration
    
    Fits gaussian distribution to phantom insert to compute mean/standard
    deviation and calibrate image intensity to density values.
    
    Inputs: path to stack of .tif images and sample name. Path to output folder.
    
    Parameters: Sigma value for the gaussian filter
    
    Output: calibration curve in .png format

"""
from distutils.util import strtobool
import argparse
import sys

from time import time
from copy import copy
from glob import glob
from os import path
import os

import findfiles

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


import SimpleITK as sitk
import numpy as np

import pandas as pd
import math

import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from pylab import *
from scipy.optimize import curve_fit

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

def getInputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--coral',
                        default='T2_PHANTOM',
                        type=str,
                        help='coral skeleton scan to analyse'
                        )
    
    parser.add_argument('-s', '--sigma',
                        default=0.001,
                        type=float,
                        help='sigma value for gaussian filter'
                        )
    
    parser.add_argument('-i', '--inp',
                        default='/Volumes/T7 Shield/PHANTOMS/T2_PHANTOM',
                        type=str,
                        help='input folder'
                        )
    parser.add_argument('-w', '--write',
                        default='/Volumes/T7 Shield/PHANTOMS/CALIBRATED',
                        type=str,
                        help='output folder'
                        )

    args = parser.parse_args()
    coral = args.coral
    inp = args.inp
    write = args.write
    sigma = args.sigma

    return coral, sigma, inp, write

def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)
    
if __name__ == "__main__":
    coral, sigma, inp, write = getInputs()
    
    
    case = glob(path.join(inp, coral))[0]
    Dir = case
    outPath = path.join(write, coral)
    # Check whether the specified path exists or not
    isExist = os.path.exists(outPath)
        
    print("reading scan ", coral)
    print('writing data to', outPath)
    
    if not isExist:
  
        # Create a new directory because it does not exist 
        os.makedirs(outPath)
        print("New output directory is created!")
        
    #-Read data stack
    print("reading stack of tiff...")
    Reader = sitk.ImageSeriesReader()
    # Find all .tif files in parent folder Dir
    name = findfiles.find_files(path.join(Dir), extension='.tif')
    for i in range(len(name)):
        name[i]=int(name[i][-8:-4])
    # Sort all images according to number (higher to lower)
    name=np.asarray(sorted(name))
    name= name[np.where(name>=0)]
    name=name[::-1]
    name = list(name)
    for i in range(len(name)):
        sN = name[i]
        name[i]=path.join(Dir, 'IMAGE' +  f"{sN:0=4d}" +'.tif')
    
    # Read image and metadata
    Reader.SetFileNames(name)
    Image = Reader.Execute()
    Spacing=Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Size = Image.GetSize()
    # Change spacing as it does not consider the axial spacing
    Spacing = [Spacing[0],Spacing[1],Spacing[0]]
    # Display image
    _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
    show_plane(a, sitk.GetArrayFromImage(Image)[Size[2] // 2], title=f'Plane = {Size[2] // 2}')
    show_plane(b, sitk.GetArrayFromImage(Image)[:, Size[1] // 2, :], title=f'Row = {Size[1] // 2}')
    show_plane(c, sitk.GetArrayFromImage(Image)[:, :, Size[0] // 2], title=f'Column = {Size[0] // 2}') 
    print("Image stack and metadata read")   
    
    # Filtering to reduce noise - here Gaussian filter
    print ("Filtering image...")  
    gaussfilt = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussfilt.SetSigma(sigma) 
    filteredImage = gaussfilt.Execute(Image)  
    # Display image
    _, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))     
    show_plane(a, sitk.GetArrayFromImage(filteredImage)[Size[2] // 2], title=f'Plane = {Size[2] // 2}')
    show_plane(b, sitk.GetArrayFromImage(filteredImage)[:, Size[1] // 2, :], title=f'Row = {Size[1] // 2}')
    show_plane(c, sitk.GetArrayFromImage(filteredImage)[:, :, Size[0] // 2], title=f'Column = {Size[0] // 2}') 
    print ("Image filtered")  
    
    # ... thresh image - we want the entire phantom but not background
    print("Thresholding image...")
    thres = sitk.TriangleThresholdImageFilter()
    # ... invert binary
    imageBinary = sitk.InvertIntensity(thres.Execute(filteredImage), maximum=1.0) 
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
    
    # Image histogram
    print("Plotting histogram")
    imData = np.ndarray.flatten(sitk.GetArrayFromImage(maskImage))
    # Select data > 0 (no background)
    imData = imData[imData!=0]
    plt.figure(figsize=(8,8))
    y,x,_=plt.hist(imData, 100, alpha=.3, label='Data')
    # First only the resin
    imDataResin = imData[imData<15000]
    # The histogram of the data
    #x, y inputs can be lists or 1D numpy arrays
    plt.figure(figsize=(8,8))
    y,x,_=plt.hist(imDataResin, 100, alpha=.3, label='Data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    # Estimation based on histogram
    expected = (10e3, 100, 5e7)
    # Curve fitting
    params, cov = curve_fit(gauss, x, y, expected)
    sigma=np.sqrt(np.diag(cov))
    x_fit = np.linspace(x.min(), x.max(), 500)
    x_fit = np.linspace(x.min(), x.max(), 500)
    # Plotting fit
    plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=3, ls="-", label='Model')
    plt.title("Resin")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=gauss.__code__.co_varnames[1:]))
    plt.show() 
    mu_resin = params[0]
    sigma_resin = params[1]
    
    # Now all the inserts - I remove the CWC fragment as it is saturated
    imDataPhantom = imData[imData>8000]
    imDataPhantom = imDataPhantom[imDataPhantom<35000]
    # The histogram of the data
    #x, y inputs can be lists or 1D numpy arrays
    plt.figure(figsize=(8,8))
    y,x,_=plt.hist(imDataPhantom, 100, alpha=.3, label='Data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    # Estimation based on histogram
        # mean of first peak, sd, peak on y-axis, (then the same for the remaining peaks)
    expected = (12500, 1000, 11e5, 17000, 4000, 85e3, 22500, 2000, 70e4)
    params, cov = curve_fit(trimodal, x, y, expected)
    sigma=np.sqrt(np.diag(cov))
    x_fit = np.linspace(x.min(), x.max(), 500)
    #plot combined...
    plt.plot(x_fit, trimodal(x_fit, *params), color='red', lw=3, label='Model')
    #...and individual Gauss curves
    plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='Distribution 1')
    plt.plot(x_fit, gauss(x_fit, *params[3:6]), color='red', lw=1, ls=":", label='Distribution 2')
    plt.plot(x_fit, gauss(x_fit, *params[6:9]), color='red', lw=1, ls="-.", label='Distribution 3')
    plt.title("Phantom")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=trimodal.__code__.co_varnames[1:]))
    plt.show() 
    mu_phantom1 = params[0]
    sigma_phantom1 = params[1]
    mu_phantom2 = params[3]
    sigma_phantom2 = params[4]
    mu_phantom3 = params[6]
    sigma_phantom3 = params[7]
    
    # Finally CWC fragment
    imDataResin = imData[imData>35000]
    imDataResin = imDataResin[imDataResin<45000]
    # The histogram of the data
    #x, y inputs can be lists or 1D numpy arrays
    plt.figure(figsize=(8,8))
    y,x,_=plt.hist(imDataResin, 100, alpha=.3, label='Data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    # Estimation based on histogram
        # mean of first peak, sd, peak on y-axis,
    expected = (39e3, 800, 12e4)
    # Curve fitting
    params, cov = curve_fit(gauss, x, y, expected)
    sigma=np.sqrt(np.diag(cov))
    x_fit = np.linspace(x.min(), x.max(), 500)
    x_fit = np.linspace(x.min(), x.max(), 500)
    # Plotting fit
    plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=3, ls="-", label='Model')
    plt.title("CWC")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=gauss.__code__.co_varnames[1:]))
    plt.show() 
    mu_cwc = params[0]
    sigma_cwc = params[1]
    
    
    # Image calibration
    d_arag = 2.94
    c_resin = 0
    c_phantom1 = 0.15
    c_phantom2 = 0.30
    c_phantom3 = 0.45
    d_resin = 1.14
    d_phantom1 = c_phantom1*d_arag + (1-c_phantom1)*d_resin
    d_phantom2 = c_phantom2*d_arag + (1-c_phantom2)*d_resin
    d_phantom3 = c_phantom3*d_arag + (1-c_phantom3)*d_resin
    d_cwc = 2.71


    calib_X = np.asarray([mu_resin, mu_phantom1, mu_phantom2, mu_phantom3]).reshape(-1, 1)
    calib_Y = np.asarray([d_resin, d_phantom1, d_phantom2, d_phantom3]).reshape(-1,1)
    df = pd.DataFrame(data=np.transpose([np.append(calib_X,mu_cwc), np.append(calib_Y,d_cwc)]), columns=['Grey value','Density in g/cm3'])
    df.to_csv(path.join(write, coral , coral + ".csv"))
    
    data_X = np.linspace(0,65534).reshape(-1, 1)
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(calib_X, calib_Y)
    
    # Make predictions using the testing set
    data_Y = regr.coef_[0][0]*data_X+regr.intercept_[0]
    # The coefficients
    print("Coefficients:")
    print("Slope: %.5g"  % regr.coef_[0][0])
    print("Intercept: %.5g" % regr.intercept_[0])
    # The mean squared error
    print("Mean squared error: %.4f" % mean_squared_error(calib_Y, regr.predict(calib_X)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.4f" % r2_score(calib_Y, regr.predict(calib_X)))
    with open(path.join(write, coral , coral +".txt"), "w") as text_file:
        text_file.write("Coefficients:\n")
        text_file.write("Slope: %.10g\n"  % regr.coef_[0][0])
        text_file.write("Intercept: %.10g\n" % regr.intercept_[0])
        text_file.write("Mean squared error: %.10f\n" % mean_squared_error(calib_Y, regr.predict(calib_X)))
        text_file.write("Coefficient of determination: %.10f\n" % r2_score(calib_Y, regr.predict(calib_X)))
        text_file.close()
    plt.figure(figsize=(8,8))
    # Plot outputs
    plt.scatter(calib_X, calib_Y, color="black", s=100)
    plt.plot(data_X, data_Y, color="blue", linewidth=3)
    plt.title(coral)
    plt.xlabel("Intensity")
    plt.ylabel(r'Density in g/cm${^3}$')
    plt.xlim([0, 65535])
    xticks(ticks=[0, 30e3, 60e3])
    plt.ylim([0, 3])
    # Add Text watermark
    text(5e3, 0.5, f'$\\rho = %.5g * I + %.5g$' % (regr.coef_[0][0], regr.intercept_[0]))   
    text(5e3, 0.2, f'$r^{2} = %.5f$' % r2_score(calib_Y, regr.predict(calib_X)))     
    plt.scatter(mu_cwc, regr.coef_[0][0]*mu_cwc+regr.intercept_[0], color='black', marker='^', s=100)
    plt.savefig(path.join(write, coral, coral +'.png'), dpi=300)
    plt.show()
    
    regr.coef_[0][0]*mu_cwc+regr.intercept_[0]
    dp1 = regr.coef_[0][0]*mu_phantom1+regr.intercept_[0]
    dp2 = regr.coef_[0][0]*mu_phantom2+regr.intercept_[0]
    dp3 = regr.coef_[0][0]*mu_phantom3+regr.intercept_[0]
    
    c22 = (dp1-d_resin)/(d_arag-d_resin)
    c33 = (dp2-d_resin)/(d_arag-d_resin)
    c44 = (dp3-d_resin)/(d_arag-d_resin)