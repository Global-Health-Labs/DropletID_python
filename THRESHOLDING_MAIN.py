# -*- coding: utf-8 -*-
"""
This is a file to optimize the thresholding of droplets
for the python version of DropletID_MAIN

Users need to change the file pathway to a folder that hold the .tif images 
to be analyzed and the sequence of images in the .tif stack 

The output  is a saved image of the circle detection, composite image (coming soon!),
 and data on the droplet signal intensity as well as thresholding the negative data

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

#import the following extra files which should be in the same folder as MAIN
from extract_Circles import  get_Circles, image_Prep, get_ROI, multiple_dfs, make_CV2image, sub_Matrix



#------Change the parameters below ----------
# path to Image folder 
directory = r'C:/Users/LaelWentland/Global Health Labs, Inc/Digital Assay Feasibility - Data/Interns/Lael W/20210723 Khe Concentration curve in droplets and bulk/Images'
replicates = 1
imagingChannels = ['BF', 'JOE']
showImage = 1  # 0 = no don't show, 1  = yes please show images!
saveToExcel = 0 # 0 = no don't save, 1 = yes please save to excel!
copiesDNA = np.array([10**5, 10**4, 10**3, 10**2, 10**1, 0], ndmin=0)* 10**(-6) # khe concentration curve
#copiesDNA = np.array([10**5, 10**5,10**5, 10**4,10**4, 10**4, 10**3,10**3,10**3, 10**2,10**2, 10**1,10**1,10**1,0,0, 0], ndmin=0)* 10**(-6)
#copiesDNA = np.array([0,25,50, 75, 100, 150], ndmin=0) # actually BP
thresholds = [.26]
#-------
#modfiy the directory path to grab all .tif files
path = directory + '/*.tif'

#move to the directory so you can save there, if you want to save elsewhere you 
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
            avgIntensity = []
            
            # call function to get ROI- see my function file for more info
            # put in the 14 bit image, not modified, the program will normalize for you
            roiIntensities = get_ROI(fluorImg, imageInfo)
            
        
            # convert roi list to array and reshape to append to the larger data storage imageInfoAll
            roiIntensities = np.array(avgIntensity)
            roiIntensities = roiIntensities.reshape((roiIntensities.shape[0], 1))
            
          
            #plot ROI intensity histogram
            plt.title(channelName + " Pixel Intensity Histogram")
            plt.xlabel("Average Normalized Intensity Value")
            plt.ylabel("Intensity Frequency")
            plt.ylim(0, 15)
            y, bins, patches = plt.hist(roiIntensities, bins = 50 , range = [0, 1], density=True)
            plt.axvline(thresholds[i-1], color='k', linestyle='dashed', linewidth=1)
            plt.show()
            
            # tack on the ROI intensity to the imageInfo
            imageInfo  = np.concatenate((imageInfo, roiIntensities), axis = 1)
            
            # thresholding to set thresholds
            posDroplets = roiIntensities > thresholds[i-1]
            imageInfoAll  = np.concatenate((imageInfo, posDroplets), axis = 1)
            
            #count number of positive droplets and save in array
            positveDroplets = np.sum(posDroplets)
            allPositive = np.append(allPositive, positveDroplets)
            
            #allBoolThresh = imageInforAll[:, imageInfoAll[:,6] == 1]
            reducedImageInfo = imageInfoAll*posDroplets
           
            # crop out the data that is not positive
            posOnlyDroplets = np.delete(reducedImageInfo,np.where(reducedImageInfo[:,6]== 0), axis=0)
             
            
            if showImage == 1:
                # modify the fluorescent/QD image to make it easier to look at
                fluorImg = exposure.rescale_intensity(fluorImg)
                img_rescale = cv2.convertScaleAbs(fluorImg, alpha =(255.0/65535.0))
                
                #make 3D so can put color on images
                img_stack = np.dstack((img_rescale,img_rescale,img_rescale))
               
                
                #draw all circles
                for i in range(imageInfoAll.shape[0]):
                    r = int(imageInfoAll[i,2])
                    x = int(imageInfoAll[i,0])
                    y = int(imageInfoAll[i,1])
                    # add circles
                    img_stack = cv2.circle(img_stack,(x,y),r, (0,255,0), thickness = 3)
                
                #draw only positive squares
                for i in range(posOnlyDroplets.shape[0]):
                    r = int(posOnlyDroplets[i,2])
                    x = int(posOnlyDroplets[i,0])
                    y = int(posOnlyDroplets[i,1])
                    
                    
                    # add rectangles
                    s = int(r / math.sqrt(2)) # 1/2 side of the square
                    startRow = int(round(x)) - s
                    startColumn = int(round(y)) - s
                    
                    #draw rectangle
                    img_stack = cv2.rectangle(img_stack,(startRow, startColumn), (startRow + s*2, startColumn + s*2), (255, 0 ,0 ), 	thickness = 2)
                
                #show the image (resized to fit screen)
                make_CV2image(img_stack, channelName + ' Circle Locations', 1,0)
                
               
            #save image info to dataframe if this is the last image in the vertical tif stack
            if channel == imagingChannels[-1]:
                allImages.append(imageInfoAll)

#%% Run gaussian fit on histogram
# show the histogram of the negative with a gaussian fit
#https://stackoverflow.com/questions/35544233/fit-a-curve-to-a-histogram-in-python
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))

from scipy.optimize import curve_fit

#grab the last few values- which should be the negatives
num_bins = 100
negIntensity = [i[:,5] for i in allImages[-replicates:]]
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
  
# save data to excel if needed
if saveToExcel == 1:
    dFImage = []
    imageNames = [i[i.find('\\')+1 : i.find('.tif')] for i in fileNames]
    # add names of channels to the columns of the sets
    imagingChannels.remove('BF')
    addString = ' Normalized Intensity'
    fluorEdit = [s + addString for s in imagingChannels]
    columnName=['x', 'y', 'Radius pixel', 'Radius um', 'Volume pL']
    columnName.extend(fluorEdit)
    columnName.extend(['Threshold Counts'])
    # convert data into data frames
    for i in allImages: 
        df = pd.DataFrame(i, columns= columnName)
        dFImage.append(df)
    
    #save data to excel sheet
    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d")
    exceFileName = directory + '/' + date + ' ImagingData.xlsx'
    # run function to make excel file
    multiple_dfs(dFImage,imageNames, 'Results', exceFileName , 1)    

# %% make array of averages for plotting  

"""
if replicates > 1:
    replicatesAll = np.array([3,3,3,2,3,3], ndmin=0) # number of replicates per sample
    startRow = 0
    for i, rep in enumerate(replicatesAll):
        subList = allImages[startRow:startRow + rep]
        startRow = startRow + rep
"""
# %% Optimize threshold against dvv

# know total droplet volume and volume of each positive component
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
    

for i in range(0, len(allEstConcDVV), 4): 
    
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


# find difference of expected vs actual
subFromActual = copiesDNA - allEstConcDVV

fig = plt.figure()
ax = plt.subplot(111) 


for i in range(0, subFromActual.shape[0], 4): 
    
    ax.plot(copiesDNA,subFromActual[i,:], 'o', label= ("%.2f" % thresholdVals[i])+ ' Threshold')
       


plt.title("DVV: Estimated Concentration - Actual DNA Concentration")
plt.xlabel('Copies of DNA copies / uL ')
plt.ylabel('Estimated - Actual')
plt.xscale('log')
plt.xlim([10**-7, 10**-.5])
plt.ylim([10**-7, 10**-.5])
plt.yscale('log')
plt.axhline(0, color='black', linestyle='-.')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
plt.show()

# %% check the submatrix info
from extract_Circles import sub_Matrix
a = np.array([1,2,3,4, 5, 6, 7, 8, 9, 10])
d = np.diag(a)

c = sub_Matrix(d, 3, 3, 4)
cm= np.mean(c)
f = sub_Matrix(d, 7, 9, 4)
fm = np.mean(f)
