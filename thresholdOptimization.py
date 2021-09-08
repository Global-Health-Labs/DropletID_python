# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:41:37 2021

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
imagingChannels = ['BF', 'JOE']
showImage = 1  # 0 = no don't show, 1  = yes please show images!
saveToExcel = 1 # 0 = no don't save, 1 = yes please save to excel!
thresholdTrue = 1 # 0 = don't do the thresholding analysis placeholder
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
    
    if len(imagingChannels) > 1: # if there is more than just BF
        # if you cant find any circles
        if imageInfo.shape == 0:
            # if you get no circles, the function should tell you but you 
            # need to stop the function
            raise Exception("No Droplets Were Detected! Check the Image Quality")
                
        #there add if
        #remove BF imgaging channel from list so it is not iterated over
        fluorQDChannels = imagingChannels.copy()
        fluorQDChannels.remove('BF')
        
        # remove the BF image from stack
        imageOtherStacks = np.copy(image)
        imageOtherStacks = np.delete(imageOtherStacks,indexBF, 2)
        
        #run through all the other stacks in the .tif file
        for i, channel in enumerate(fluorQDChannels):
            channelName = channel + ' in '  + imageName
            
            #modify images for analysis
            org_img = imageOtherStacks[:,:,i]
           
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
                posDropletsBool = avgRoiVal >= thresholds[i]
                # add data to the image info matrix
                imageInfo  = np.concatenate((imageInfo, posDropletsBool), axis = 1)
                
                
                #plot ROI intensity histogram with cutoff line
                plt.title(channelName + " Pixel Intensity Histogram")
                plt.xlabel("Average Normalized Intensity Value")
                plt.ylabel("Intensity Frequency")
                plt.ylim(0, 20)
                y, bins, patches = plt.hist(avgRoiVal, bins = 50 , range = [0, 1], density=True)
                plt.axvline(thresholds[i], color='k', linestyle='dashed', linewidth=1)
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
            if channel ==  fluorQDChannels[-1]:
                allImages.append(imageInfo)
    else:
        # save information to image Info even if the image only contains the BF
       allImages.append(imageInfo)
           
           
    # create  and save composite images if needed
    #if compositeSave == 1:
     # make composite image: https://stackoverflow.com/questions/65439230/convert-grayscale-2d-numpy-array-to-rgb-image

volDistribution = [img[:,-3] for img in allImages]

# save data to excel if needed
if saveToExcel == 1:
    dFImage = []
    imageNames = [i[i.rindex('\\')+1 : i.find('.tif')] for i in fileNames]
    # add names of channels to the columns of the sets
    
    addString = ' Normalized Intensity'
    fluorEdit = [s + addString for s in fluorQDChannels]    
    columnName=['x', 'y', 'Radius pixel', 'Radius um', 'Volume pL']
    
    if len(imagingChannels) > 1:
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

#%% Can use to check your threshold
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))

from scipy.optimize import curve_fit

#grab the last few values- which should be the negatives
num_bins = 1000
negIntensity = [i[:,5] for i in allImages[-3:]]
thresholdAll = [];

for column in negIntensity:
    currentNeg = column
    mu = np.mean(currentNeg)
    sigma = np.std(currentNeg)
    
    # The histogram of the data.
    plt.figure(1)
    yvalsN, binsN, patchesN = plt.hist(currentNeg, num_bins, range = [0, 1], density = True , label='Negative data')
    bin_center = binsN[:-1] + np.diff(binsN) / 2
    popt,pov = curve_fit(gaussian, bin_center, yvalsN)
    
    std3 = popt[0] + 4*popt[2]
    
    x_interval_for_fit = np.linspace(binsN[0], binsN[-1], 10000)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='fit')
    gaussianVals = gaussian(x_interval_for_fit, *popt)
    plt.axvline(std3, color='k', linestyle='dashed', linewidth=1)
    plt.text(std3+.1, 20, r'Threshold 4$\sigma$ : %.2f' %std3)
    plt.text(std3+.1, 10, r' Peak x val : %.2f' %popt[0])
    plt.text(std3+.1,5, r'1$\sigma$ : %.3f' %popt[2])
    plt.legend()
    plt.xlabel('Avg ROI Normalized Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Negative Droplet Intensity')         
    plt.show()
    thresholdAll.append(std3)

meanThresh= np.mean(thresholdAll)
stdThresh =np.std(thresholdAll)
print(r'Average Threshold: %.3f +/- %.3f' % (meanThresh, stdThresh) )  
# %% can use to optimize threshold
copiesDNA = np.array([10**5, 10**4, 10**3, 10**2, 10**1, 0], ndmin=0)* 9.01*10**(-6) # khe concentration curve
os.chdir('C:/Users/LaelWentland/PythonImageAnalysis')
from inference import lam_dvv


#list of threshold values to test
thresholdVals = np.arange(0.1,0.5,0.01)
# list of only the volumes of the data
volData =[arr[:, 4] for arr in allImages]
#list of only the normalized data
normData =[arr[:, 5] for arr in allImages]

def getPosVol(params, intensityList):
    #  implement model
    alpha = params  # threshold
    allCount = np.array([])
    posVolList = []
    
    for i in intensityList[:]:
        posVolInd = np.where(i >= alpha )
        count = len(posVolInd)
        # add count to np array
        allCount = np.append(allCount, [count], axis = 0)
        # add indicies to list
        posVolList.append(allCount)
        
    return posVolList

allEstConcDVV = []

for thresh in thresholdVals:
    allEstConc = np.array([])
    posVolList = []
    #volListIndex = getPosVol(thresh, normData)
    alpha = thresh
    
    for w in normData:
        posVolInd = np.where(w >= alpha )
        posInd = np.asarray(posVolInd)
        posVolList.append(posInd)
    
    # loop throuh list of the positive indicies
    for i, ind in enumerate(posVolList):
        currentCol = volData[i]
        volPos = currentCol[ind]
        getEstCon = lam_dvv(volPos, currentCol.sum())
        
        allEstConc = np.append(allEstConc,  getEstCon)
       
        
    allEstConcDVV.append(allEstConc)
#%% plot replicates
stackDVVestConc  = np.stack(allEstConcDVV, axis=0)
fig = plt.figure()
ax = plt.subplot(111) 

# copies of DNA/ pL
    

for i in range(6, len(allEstConcDVV), 4): 
    
    ax.plot(copiesDNA,allEstConcDVV[i], 'o', label= ("%.2f" % thresholdVals[i])+ ' Threshold')
       


plt.title("DVV: Estimated Concentration vs Copies of DNA added")
plt.xlabel('Copies of DNA copies / uL ')
plt.ylabel('Estimated Concentration DVV copies/ pL')
plt.xscale('log')
plt.xlim([10**-7, 10**-.5])
plt.ylim([10**-7, 10**-.5])
plt.yscale('log')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
plt.show()

#%%

# find difference of expected vs actual
subFromActual = copiesDNA - allEstConcDVV

fig = plt.figure()
ax = plt.subplot(111) 

#copiesDNA = copiesDNA[:-1]
for i in range(0, subFromActual.shape[0], 4): 
    
    ax.plot(copiesDNA,subFromActual[i,:], 'o', label= ("%.2f" % thresholdVals[i])+ ' Threshold')
    """
    m, c = np.polyfit(np.log(copiesDNA), np.log(subFromActual[i,:-1]), 1) # fit log(y) = m*log(x) + c
    xRange = np.arange(1, np.max(copiesDNA),100)
    y_fit = np.exp(m*np.log(copiesDNA) + c) # calculate the fitted values of y in linear space    
    ax.plot(xRange, y_fit, '-')
    # r-squared calculations
    p = m*np.log(copiesDNA) + c # in log-log space
    # fit values, and mean
    yhat = p(np.log(copiesDNA))                         
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    r2 = ssreg / sstot
    print(r2)
    """
plt.title("DVV: Estimated Concentration - Actual DNA Concentration")
plt.xlabel('Copies of DNA copies / uL ')
plt.ylabel('Estimated - Actual')
plt.xscale('log')
#plt.xlim([10**-7, 10**-.5])
#plt.ylim([0, 10**-2])
#plt.yscale('log')
plt.axhline(0, color='black', linestyle='-.')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
plt.show()