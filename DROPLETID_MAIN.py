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
import pandas as pd # for making excel sheet
import datetime


#import the following extra files which should be in the same folder as MAIN
from extract_Circles import  get_Circles, image_Prep, get_ROI, multiple_dfs, make_CV2image, determineThreshold
from inference import lam_dvv
#%%
#------Change the parameters below ----------

# path to Image folder - copy absolute path of the folder
directory = r'C:/Users/LaelWentland/Global Health Labs, Inc/Digital Assay Feasibility - Data/Interns/Lael W/20210826 Louise MultiModal Data/20210824Images'
replicates = 5
noOfNegatives = 4 # just in case you have different number of negative images than regular replicates
imagingChannels = ['Cy5', 'FAM', 'BF'] # list in order the image acquisition mode
showImage = 0  # 0 = no don't show, 1  = yes please show images!
saveToExcel = 1 # 0 = no don't save, 1 = yes please save to excel!
thresholdTrue = 1 # 0 = don't do the thresholding analysis placeholder
compositeSave = 0 # 0 = don't create and show composite images- coming soon!
thresholds = { 'FAM': 0.83, 'Cy5': 0.6} # choose threshold manually if you want, 
#if thresholds dictionary is left blank then the computer will automatically grab the last few 
# images and generate a threshold based on those
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
allImageData = []
allImages = []
allImageNames = []
#Array that stores the number of positive droplets found
allPositive = np.array([])

# run through all images and grab data
for count, name in enumerate(fileNames):
    #tifffile-used to open tif images
    #image should alrady be in grey scale so no need to change to RGB space
    image = tifffile.imread(name)
    allImages.append(image)
    imageName = name[name.rindex('\\')+1:name.find('.tif')]
    allImageNames.append(imageName)
    
    # get location of brightfield in image stack
    indexBF = imagingChannels.index('BF')
    imageBF = image[:,:,indexBF]
    
    # adjust the BF image to increase contrast and move to 8 bit space
    # in order to extract the circles
    imageAdjusted = image_Prep(imageBF, 'BF', imageName, 0)
    
    #extract circle info from images
    imageInfo = get_Circles(imageAdjusted, 'BF in ' + imageName, showImage)
    if imageInfo.shape == 0:
        # if you get no circles, the function should tell you but you 
        # need to stop the function
        raise Exception("No Droplets Were Detected! Check the Image Quality")
    allImageData.append(imageInfo)

 # if there is more than just BF then analyze the other chnanels
#%%    

# look at Other channels and analyze  
if len(imagingChannels) > 1:
    
    #get indicies of things that are not BF
    nonBFIndex = [i for i in range(len(imagingChannels)) if imagingChannels[i] != 'BF']
    
    
    # establish thresholds for each channel    
    for channel in nonBFIndex:
        #Grab the negative files and determine thresholds if has not been set
        
        if thresholdTrue == 1 and (imagingChannels[channel] in thresholds) == False:
            
            pixelData = []
            # run through the last few images, which shoulw be negative
            # generate a threshold that is 4 standard deviations from average peak
            for image, iInfo in zip(allImages[-noOfNegatives:], allImageData[-noOfNegatives:]):
                roiLoc, avgRoiVal = get_ROI(image[:,:,channel], iInfo, rects_scalefactor)
                avgRoiVal =  avgRoiVal.reshape((avgRoiVal.shape[0], 1))
                pixelData.append(avgRoiVal)
            
            # plug in roi average intensity of 
            avgThresh = determineThreshold(pixelData, imagingChannels[channel])
            # make threshold in threshold dictionary
            thresholds[imagingChannels[channel]] = avgThresh
            
    
    # determine the ROIs in the region and threshold data 
    for num, image in enumerate(allImages):
        allPositiveDrop = np.zeros((allImageData[num].shape[0],len(nonBFIndex)))
        
        for channel in nonBFIndex:
            channelName = imagingChannels[channel] + ' in '  + allImageNames[num]
            
            #select images for analysis
            org_img = image[:,:,channel]
           
            fluorImg = org_img
            img_max = np.array(fluorImg.shape).max()
            img_min = 0
            
            # run function to  get average normalized ROI value
            roiLoc, avgRoiVal = get_ROI(fluorImg, allImageData[num], rects_scalefactor)
            avgRoiVal =  avgRoiVal.reshape((avgRoiVal.shape[0], 1))
            # add ROI average value to image info matrix
            allImageData[num]  = np.concatenate((allImageData[num], avgRoiVal), axis = 1)
            
            if thresholdTrue == 1:
                
                # make array of boolean for the average values
                posDropletsBool = avgRoiVal >= thresholds[imagingChannels[channel]]
                
                allPositiveDrop[:,channel] =  posDropletsBool.ravel()
                # add data to the image info matrix
                allImageData[num]  = np.concatenate((allImageData[num], posDropletsBool), axis = 1)
                
                
                #plot ROI intensity histogram with cutoff line
                plt.title(channelName + " Pixel Intensity Histogram")
                plt.xlabel("Average Normalized Intensity Value")
                plt.ylabel("Intensity Frequency")
                plt.ylim(0, 20)
                y, bins, patches = plt.hist(avgRoiVal, bins = 50 , range = [0, 1], density=True)
                plt.axvline( thresholds[imagingChannels[channel]], color='k', linestyle='dashed', linewidth=1)
                plt.show()
                
                #show rois on image
                img_rescale= exposure.rescale_intensity(fluorImg)
                img_rescale = cv2.convertScaleAbs(img_rescale, alpha =(255.0/65535.0))
                #make 3D so can put color on images
                img_overlay = np.dstack((img_rescale,img_rescale,img_rescale))
                # i is the 0 or 1 if the droplet is positive, circs contains the (x, y, radius in pixels)
                # rect is for the locaitons to draw the rectangles
                for i, circ, rect in zip(posDropletsBool, allImageData[num][:,0:3], roiLoc):
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
            if channel ==  nonBFIndex[-1]:
                # determine droplets that are positive for both fluor channels
                bothPositive = np.all( allPositiveDrop, axis = 1 )
                allImageData[num] = np.concatenate((allImageData[num], bothPositive.reshape(bothPositive.shape[0], 1)),axis = 1)

          
# create  and save composite images if needed
#if compositeSave == 1:
# make composite image: https://stackoverflow.com/questions/65439230/convert-grayscale-2d-numpy-array-to-rgb-image

volDistribution = [img[:,-3] for img in allImageData]
#%%
# save data to excel if needed
if saveToExcel == 1:
    dFImage = []
    imageNames = [i[i.rindex('\\')+1 : i.find('.tif')] for i in fileNames]
    # add names of channels to the columns of the sets
    
    addString = ' Normalized Intensity'
    channelEdit = [imagingChannels[i] for i in nonBFIndex]
    fluorEdit = [s + addString for s in channelEdit]
    columnName=['x', 'y', 'Radius pixel', 'Radius um', 'Volume pL']
    
    if len(imagingChannels) > 1:
        if thresholdTrue == 1:
            # edit the column names 
            posName = 'Positive Droplets'
            
            fluorEdit = [e for i in fluorEdit for e in [i, i.replace(' Normalized Intensity', " ") + posName ]]
            columnName.extend(fluorEdit)
            
            columnName.extend(['All Positive'])
        else:
            columnName.extend(fluorEdit)
    
    # convert data into data frames
    for i in allImageData: 
        df = pd.DataFrame(i,columns= columnName )
        dFImage.append(df)
    
    #save data to excel sheet
    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d")
    exceFileName = directory + '/' + date + ' ImagingData.xlsx'
    multiple_dfs(dFImage, imageNames, 'Results', exceFileName , 1)    

#%% Use to check stats of data with DVV
from collections import defaultdict
#convert from pL to uL for all the volumes
totalDroplets =  [i[:, 4]*10**-6 for i in allImageData]
totalPositive = [i[:, 8]*i[:,4]*10**-6   for i in allImageData] # for IL8
#totalPositive = [i[:, 6]*i[:,4]*10**-6   for i in allImageData] # for gDNA
totalPositive = [np.delete(arr, np.argwhere( (arr == 0))) for arr in totalPositive]



#dnaCurve = {'A' :1000,'B': 100, 'C': 10, 'D' : 5, 'E': 1,'F': 0.5,'G': 0.1,  'H': 0} # copies/uL
estConcDVV = defaultdict(list)
"""
for num, tot, pos in zip(allImageNames, totalDroplets, totalPositive):
   
    estConc =  lam_dvv(pos, tot.sum())
    estConcDVV[num[:1]].append(estConc)


for key in dnaCurve:
    x = np.ones(len( estConcDVV[key]))*dnaCurve[key] 
    plt.scatter(x, estConcDVV[key])
    
plt.title("DVV Estimated vs Input gDNA kHE concentration")
plt.xlabel('Concentration of gDNA (copies/uL)')
plt.ylabel('Estimated gDNA Concentration (copies/uL)')
plt.yscale('log')
plt.xscale('log')
plt.xlim([10**-2, 10**6])
plt.ylim([10**-2, 10**6])
plt.show()
"""

iL8Curve = {'A' :0.1,'B': 0.05, 'C': 0.01, 'D' : 0.005, 'E': 0.001,'F': 0.0005,'G': 0.0001,  'H': 0} # pg/uL, was in pg/mL
estConcDVV = defaultdict(list)
for num, tot, pos in zip(allImageNames, totalDroplets, totalPositive):
    
    estConc =  lam_dvv(pos, tot.sum())
    estConcDVV[num[:1]].append(estConc)


for key in iL8Curve:
    x = np.ones(len( estConcDVV[key]))*iL8Curve[key] 
    plt.scatter(x, estConcDVV[key])
plt.title("DVV Estimated vs Input IL8 concentration")
plt.xlabel('Concentration of IL8 (pg/uL)')
plt.ylabel('Estimated IL8 Concentration (pg/uL)')
plt.yscale('log')
plt.xlim([10**-5, 10**4])
plt.ylim([10**-5, 10**4])
plt.xscale('log')
plt.show()
