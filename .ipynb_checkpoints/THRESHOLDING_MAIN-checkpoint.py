# -*- coding: utf-8 -*-
"""
This is a file to optimize the thresholding of droplets based on negative
controls for the python version of DropletID_MAIN

Users need to change the file pathway to a folder that hold the .tif images 
to be analyzed and the sequence of images in the .tif stack 

**IMPORTANT**The negatives should be the last couple of images in the 
list, if not you can modify the images names list at the top

@author: LaelWentland
"""
#import necessary toolboxes
import cv2         #openCV
import numpy as np # super necessary
import tifffile    # to read stacked tiff file
import glob        # to grab all files in folder
import os          # to help move you in the correct directory
import math
from matplotlib import pyplot as plt # for plotting
from skimage import exposure        # for increasing brightness, rescales image
from PIL import Image, ImageDraw, ImageFilter # for making composite image
import pandas as pd # need for converting to excel
import datetime     # need for saving to excel
from scipy.optimize import curve_fit

#import the following extra files which should be in the same folder as MAIN
from extract_Circles import  get_Circles, image_Prep, get_ROI, multiple_dfs, make_CV2image, sub_Matrix

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))


#------Change the parameters below ----------
# path to Image folder 
directory = r'C:/Users/LaelWentland/Global Health Labs, Inc/Digital Assay Feasibility - Data/Interns/Lael W/20210723 Khe Concentration curve in droplets and bulk/Images'
replicates = 1
imagingChannels = ['BF', 'JOE']
showImage = 0  # 0 = no don't show, 1  = yes please show images!
#-------
#modfiy the directory path to grab all .tif files
path = directory + '/*.tif'

#move to the directory so you can save there, if you want to save elsewhere you 
#can change the line below to move to the path of your chosing
os.chdir(directory)

#grabs all the file names of .tif images in folder
fileNames = glob.glob(path)
fileNames.sort()

#-- change  this if needed---
# only grab the last few images that are negatives
fileNames = fileNames[-replicates:]
#List that will store thresholds
thresholdAll = []


for count, name in enumerate(fileNames):
    #tifffile- seems to work 
    #image should alrady be in grey scale so 
    image = tifffile.imread(name)
    imageName = name[name.find('\\')+1:name.find('.tif')]
    
    #run through all the stacks in the .tif file
    for i, channel in enumerate(imagingChannels):
        channelName = channel + ' in '  + imageName
        
        #modify images for analysis
        org_img = image[:,:,i]
        if channel == 'BF':
            
            # adjuste the BF image to increase contrast and move to 8 bit space
            # in order to extract the circles
            imageAdjusted = image_Prep(org_img, channel, imageName, showImage)
            #extract circles from images
            imageInfo = get_Circles(imageAdjusted, channelName, showImage)
            # if you cant find
            if imageInfo.shape == 0:
                # if you get no circles, the function should tell you but you 
                # need to stop the function
                raise Exception("No Droplets Were Detected! Check the Image Quality")
            
        else:
            fluorImg = org_img
            alpha = 3 # Contrast control (1.0-3.0)
            beta = 3 # Brightness control (0-100)
            fluorImgScale = cv2.convertScaleAbs(fluorImg, alpha=alpha, beta=beta)
         
            img_rescale = cv2.convertScaleAbs(fluorImgScale, alpha =(255.0/65535.0))
            img_stack = np.dstack((img_rescale,img_rescale,img_rescale))
            normFluorImg = fluorImg / (2**14)
            
            
            # call function to get ROI- see my function file for more info
            # put in the 14 bit image, not modified, the program will normalize for you
            negIntensity = get_ROI(fluorImg, imageInfo)
            
           
            #negIntensity = roiIntensities.reshape((roiIntensities.shape[0], 1))
            
            #grab the last few values- which should be the negatives
            num_bins = 100
            mu = np.mean(negIntensity)
            sigma = np.std(negIntensity)
            
            # The histogram of the data fitted to gaussian curve
            # Code adapted from the link below
            #https://stackoverflow.com/questions/35544233/fit-a-curve-to-a-histogram-in-python

            plt.figure(i+1)
            yvalsN, binsN, patchesN = plt.hist(negIntensity, num_bins, range = [0, 1], density = True , label='Negative data')
            bin_center = binsN[:-1] + np.diff(binsN) / 2
            popt,pov = curve_fit(gaussian, bin_center, yvalsN)
            
            std3 = popt[0] + 3*popt[2]
            
            x_interval_for_fit = np.linspace(binsN[0], binsN[-1], 10000)
            plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='fit')
            gaussianVals = gaussian(x_interval_for_fit, *popt)
            plt.axvline(std3, color='k', linestyle='dashed', linewidth=1)
            plt.text(std3+.1, 8, r'Threshold 3$\sigma$ : %.2f' %std3)
            plt.text(std3+.1, 7, r' Peak x val : %.2f' %popt[0])
            plt.text(std3+.1,6, r'1$\sigma$ : %.3f' %popt[2])
            plt.legend()
            plt.xlabel('Avg ROI Normalized Intensity')
            plt.ylabel('Frequency')
            plt.title('Histogram of Negative Droplet Intensity')         
            plt.show()
            thresholdAll.append(std3)
        
        meanThresh= np.mean(thresholdAll)
        stdThresh =np.std(thresholdAll)
        print(r'Average Threshold: %.3f +/- %.3f' % (meanThresh, stdThresh) ) 
          

  