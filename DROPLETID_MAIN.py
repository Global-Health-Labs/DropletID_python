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
directory = r'C:/Users/LaelWentland/Global Health Labs, Inc/Digital Assay Feasibility - Data/Interns/Lael W/20210723 Khe Concentration curve in droplets and bulk/Images'
replicates = 1
imagingChannels = ['BF', 'JOE']
showImage = 1  # 0 = no don't show, 1  = yes please show images!
saveToExcel = 1 # 0 = no don't save, 1 = yes please save to excel!
thresholdTrue = 1 # 0 = don't do the thresholding analysis placeholder
compositeSave = 0 # 0 = don't create and show composite images- coming soon!
thresholds = [.26] # choose threshold manually, I may change to automatic later, based on neg data

#------------------------------------

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
            
            # adjuste the BF image to increase contrast and move to 8 bit space
            # in order to extract the circles
            # function below
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
           
            # run function to  get ROI average intensity in the squares
            roiIntensities = get_ROI(fluorImg, imageInfo)
            # convert data to np array
            roiNParray= np.array(roiIntensities)
            roiNParray = roiNParray.reshape((roiNParray.shape[0],1))
            imageInfoAll  = np.concatenate((imageInfo, roiNParray), axis = 1)
            
            # threshold data
            if thresholdTrue == 1:
                posDropletsBool = roiNParray >= thresholds[i-1]
                
                # add boolean array to all imageInfo
                imageInfoAll  = np.concatenate((imageInfoAll, posDropletsBool), axis = 1)
                
                # get total number of positive droplets
                positveDropletsSum = np.sum(posDropletsBool)
                allPositive = np.append(allPositive, positveDropletsSum)
                
                #plot ROI intensity histogram with cutoff line
                plt.title(channelName + " Pixel Intensity Histogram")
                plt.xlabel("Average Normalized Intensity Value")
                plt.ylabel("Intensity Frequency")
                plt.ylim(0, 15)
                y, bins, patches = plt.hist(roiIntensities, bins = 50 , range = [0, 1], density=True)
                plt.axvline(thresholds[i-1], color='k', linestyle='dashed', linewidth=1)
                plt.show()
                
            
            # draw keypoints onfluorescent images
            if showImage == 1:
                currentImageData = imageInfoAll
                
                # modify the fluorescent/QD image to make it easier to look at
                fluorImg = exposure.rescale_intensity(fluorImg)
                img_rescale = cv2.convertScaleAbs(fluorImg, alpha =(255.0/65535.0))
                
                #make 3D so can put color on images
                img_stack = np.dstack((img_rescale,img_rescale,img_rescale))
                
                if thresholdTrue == 1:
                    # only plot/outline the positive droplets
                    reducedImageInfo = currentImageData*posDropletsBool
           
                    # crop out the data that is not positive
                    currentImageData = np.delete(reducedImageInfo,np.where(reducedImageInfo[:,6]== 0), axis=0)
             
                
                for i in range(imageInfoAll.shape[0]):
                    r = int(imageInfoAll[i,2])
                    x = int(imageInfoAll[i,0])
                    y = int(imageInfoAll[i,1])
                    
                    # add circles to all detected cirlces
                    img_rescale = cv2.circle(img_stack,(x,y),r, (0,255,0), thickness = 3)
                    
                for i in range(currentImageData.shape[0]):
                    r = int(currentImageData[i,2])
                    x = int(currentImageData[i,0])
                    y = int(currentImageData[i,1])
                    
            
                    s = int(r / math.sqrt(2)) # 1/2 side of the square
                    startRow = int(round(x)) - s
                    startColumn = int(round(y)) - s
                    
                    #draw rectangles on only detected positive droplets
                    modImage = cv2.rectangle(img_stack,(startRow, startColumn), (startRow + s*2, startColumn + s*2), (255,0,0), 	thickness = 2)
                
                
                #show the image (resized to fit screen)
                make_CV2image(img_stack, channelName + ' Circle Locations', 1,0)
                
                
            
            #save image info to List if this is the last image in the vertical tif stack
            if channel == imagingChannels[-1]:
                allImages.append(imageInfoAll)
    
    # create  and save composite images if needed
    #if compositeSave == 1:
     # make composite image: https://stackoverflow.com/questions/65439230/convert-grayscale-2d-numpy-array-to-rgb-image
   

# save data to excel if needed
if saveToExcel == 1:
    dFImage = []
    imageNames = [i[i.find('\\')+1 : i.find('.tif')] for i in fileNames]
    # add names of channels to the columns of the sets
    imagingChannels = imagingChannels.remove('BF')
    addString = ' Normalized Intensity'
    fluorEdit = [s + addString for s in imagingChannels]    
    columnName=['x', 'y', 'Radius pixel', 'Radius um', 'Volume pL']
    columnName.extend(fluorEdit)
    
    if thresholdTrue == 1:
        columnName.extend('Positive Droplets')
        
    # convert data into data frames
    for i in allImages: 
        df = pd.DataFrame(i,columns= columnName )
        dFImage.append(df)
    
    #save data to excel sheet
    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d")
    exceFileName = directory + '/' + date + ' ImagingData.xlsx'
    multiple_dfs(dFImage, imageNames, 'Results', exceFileName , 1)    


