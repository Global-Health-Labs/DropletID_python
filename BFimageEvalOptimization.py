# -*- coding: utf-8 -*-
"""
Code to try different options for improving the BF detection

@author: LaelWentland
"""

#import necessary toolboxes

import cv2         #openCV
import numpy as np
import tifffile    # to read stacked tiff file
import glob        # to grab all files in folder
import os          # to help move you in the correct directory
import math
from matplotlib import pyplot as plt # for plotting
from skimage import exposure        # for increasing brightness, rescales image
from PIL import Image, ImageDraw, ImageFilter # for making composite image
import pandas as pd
import datetime

#import the following extra files which should be in the same folder as MAIN
from extract_Circles import  get_Circles, image_Prep, get_ROI, multiple_dfs, make_CV2image
from inference import lam_dvv

#------Change the parameters below ----------

# path to Image folder - copy absolute path of the folder
directory = r'C:/Users/LaelWentland/Global Health Labs, Inc/Digital Assay Feasibility - Data/Interns/Lael W/20210723 Khe Concentration curve in droplets and bulk/Images'
replicates = 1
imagingChannels = ['BF', 'JOE'] # list in order the image acquisition mode
showImage = 1  # 0 = no don't show, 1  = yes please show images!
saveToExcel = 0 # 0 = no don't save, 1 = yes please save to excel!
thresholdTrue = 0 # 0 = don't do the thresholding analysis placeholder
compositeSave = 0 # 0 = don't create and show composite images- coming soon!
thresholds = [.26] # choose threshold manually, I may change to automatic later, based on neg data
# there should be one threshold for each channel that is not BF
#------------------------------------

# the scale factor for how small the ROI square should be in the droplet
rects_scalefactor = 0.9

#modfiy the directory path to grab all .tif files
path = directory + '/*.tif'

#moves you to the directory so you can save there, if you want to save elsewhere you 
#can change the line below to move to the path of your chosing
os.chdir(directory)

#grabs all the file names of .tif images in folder
fileNames = glob.glob(path)
fileNames.sort()

#List that will store all the data on droplet locations and intensities
allImages = []

#Array that stores the number of positive droplets found
allPositive = np.array([])

for count, name in enumerate(fileNames):
    #tifffile- seems to work 
    #image should alrady be in grey scale so no need to change to RGB space
    image = tifffile.imread(name)
    imageName = name[name.rindex('\\')+1:name.find('.tif')]
    
    # get location of brightfield in image stack
    indexBF = imagingChannels.index('BF')
    imageBF = image[:,:,indexBF]
    
    # adjust the BF image to increase contrast and move to 8 bit space
    # in order to extract the circles
    imageAdjusted = image_Prep(imageBF, 'BF', imageName, showImage)
    
    #extract circle info from images
    imageInfo = get_Circles(imageAdjusted, 'BF in ' + imageName, showImage)
    allImages.append(imageInfo)
    
    