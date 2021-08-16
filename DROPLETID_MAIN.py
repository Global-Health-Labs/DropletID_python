# -*- coding: utf-8 -*-
__author__ = 'Lael Wentland'
__copyright__ = 'Copyright 2021, GH labs'
__credits__ = ['Lael Wentlad', 'Louise Hansen', 'Josh Bishop']
__license__ = 'Apache 2.0'
__version__ = '1.0'
__maintainer__ = 'Lael Wentland'
__email__ = 'lael.wentland@ghlabs.org'
__status__ = 'Development'
__fullname__ = 'Digital Droplets Project'

"""
This is the main file for the python version of DropletID

Users need to change the file pathway to a folder that hold the .tif images 
to be analyzed and the sequence of images in the .tif stack

If you choose to show images you need to exit each image that pops up in the viewer 
or else the code wont go on, 

The output  is a saved image of the circle detection, composite image (coming soon)
 and data on the number of positive droplets

Necessary files to be in folder in addition to all the different libraries I use: 
    1) extract_Circle.py : holds all the functions for image analysis
    2) inference.py : stats analysis files to get concentration

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


#------Change the parameters below ----------

# path to Image folder - copy absolute path of the folder
directory = r'C:/Users/LouiseHansen/Global Health Labs, Inc/Digital Assay Feasibility - Louise H/08062021 - imaging of multimodal right after assay completion'
replicates = 1
imagingChannels = ['Cy5', 'BF', 'JOE']
showImage = 0  # 0 = no don't show, 1  = yes please show images!
saveToExcel = 1 # 0 = no don't save, 1 = yes please save to excel!
thresholdTrue = 1 # 0 = don't do the thresholding analysis placeholder
compositeSave = 0 # 0 = don't create and show composite images- coming soon!
thresholds = [.3, .3] # choose threshold manually, I may change to automatic later, based on neg data
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
    imageName = name[name.find('\\')+1:name.find('.tif')]
    
    #run through all the stacks in the .tif file
    for i, channel in enumerate(imagingChannels):
        channelName = channel + ' in '  + imageName
        
        #modify images for analysis
        org_img = image[:,:,i]
        if channel == 'BF':
            
            # adjust the BF image to increase contrast and move to 8 bit space
            # in order to extract the circles
            imageAdjusted = image_Prep(org_img, channel, imageName, showImage)
            
            #extract circle info from images
            imageInfo = get_Circles(imageAdjusted, channelName, showImage)
            
            # if you cant find any circles
            if imageInfo.shape == 0:
                # if you get no circles, the function should tell you but you 
                # need to stop the function
                raise Exception("No Droplets Were Detected! Check the Image Quality")
            
        else:
            # image should not be modified
            fluorImg = org_img
            img_max = np.array(fluorImg.shape).max()
            img_min = 0
            
            # run function to  get average normalized ROI value
            roiLoc, avgRoiVal = get_ROI(fluorImg, imageInfo, rects_scalefactor)
            avgRoiVal =  avgRoiVal.reshape((avgRoiVal.shape[0], 1))
            # add ROI average value to image info matrix
            imageInfo  = np.concatenate((imageInfo, avgRoiVal), axis = 1)
            
            if thresholdTrue == 1:
                
                # make array of boolean for the average values
                posDropletsBool = avgRoiVal >= thresholds[i-1]
                # add data to the image info matrix
                imageInfo  = np.concatenate((imageInfo, posDropletsBool), axis = 1)
                
                
                #plot ROI intensity histogram with cutoff line
                plt.title(channelName + " Pixel Intensity Histogram")
                plt.xlabel("Average Normalized Intensity Value")
                plt.ylabel("Intensity Frequency")
                plt.ylim(0, 15)
                y, bins, patches = plt.hist(avgRoiVal, bins = 50 , range = [0, 1], density=True)
                plt.axvline(thresholds[i-1], color='k', linestyle='dashed', linewidth=1)
                plt.show()
                
                #show rois on image
                img_rescale= exposure.rescale_intensity(fluorImg)
                img_rescale = cv2.convertScaleAbs(img_rescale, alpha =(255.0/65535.0))
                #make 3D so can put color on images
                img_overlay = np.dstack((img_rescale,img_rescale,img_rescale))
                # i is the 0 or 1 if the droplet is positive, circs contains the (x, y, radius in pixels)
                # rect is for the locaitons to draw the rectangles
                for i, circ, rect in zip(posDropletsBool, imageInfo[:,0:3], roiLoc):
                    # determine if the average pixel value in the rectangular roi is larger than the threshold
                    if i == 1:
                        circ_color = (0, 255, 0) # draw green
                    else:
                        circ_color = (255, 0, 0) # draw red
                    img_overlay = cv2.circle(img_overlay, (int(circ[0]), int(circ[1])), int(circ[2]), circ_color, 2)
                    #img_overlay = cv2.rectangle(img_overlay,(rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[2]), circ_color, 2)
            
                if showImage == 1:
                   
                    make_CV2image(img_overlay, channelName + ' Circle Locations', 1,1)
                else:
                    make_CV2image(img_overlay, channelName + ' Circle Locations', 0,1)
                
            
            #save image info to List if this is the last image in the vertical tif stack
            if channel == imagingChannels[-1]:
                allImages.append(imageInfo)
    
    # create  and save composite images if needed
    #if compositeSave == 1:
     # make composite image: https://stackoverflow.com/questions/65439230/convert-grayscale-2d-numpy-array-to-rgb-image


# save data to excel if needed
if saveToExcel == 1:
    dFImage = []
    imageNames = [i[i.find('\\')+1 : i.find('.tif')] for i in fileNames]
    # add names of channels to the columns of the sets
    
    addString = ' Normalized Intensity'
    fluorEdit = [s + addString for s in imagingChannels[1:]]    
    columnName=['x', 'y', 'Radius pixel', 'Radius um', 'Volume pL']
    
    
    if thresholdTrue == 1:
        # edit the column names 
        posName = 'Positive Droplets'
        
        fluorEdit = [e for i in fluorEdit for e in [i, i.replace(' Normalized Intensity', " ") + posName ]]
        columnName.extend(fluorEdit)
    else:
        columnName.extend(fluorEdit)
    
    # convert data into data frames
    for i in allImages: 
        df = pd.DataFrame(i,columns= columnName )
        dFImage.append(df)
    
    #save data to excel sheet
    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d")
    exceFileName = directory + '/' + date + ' ImagingData.xlsx'
    multiple_dfs(dFImage, imageNames, 'Results', exceFileName , 1)    


