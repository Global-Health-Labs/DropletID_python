# -*- coding: utf-8 -*-
"""
Fucntions used for extracting the circles in an image

@author: LaelWentland
"""
import cv2
import numpy as np
import pandas as pd
from skimage import exposure
import math


# internal function that grabs a smaller section of a larger matrix
def sub_Matrix(matrix, startRow, startCol, size):
    return matrix[startRow:startRow+size,startCol:startCol+size]

# This shortens the produciton and saving of CV2 images
def make_CV2image(image,imageTitle, showImage, saveImage):
    
    #function resizes image to smaller
    re = 600.0 / image.shape[1]
    dim = (600, int(image.shape[0] * re))
    
    resizeOutput = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if showImage == 1:
        cv2.imshow(imageTitle, resizeOutput )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if saveImage == 1:
        imageTitle =  imageTitle + '.png'
        cv2.imwrite(imageTitle, image)

# callable function that takes in a BF image and finds the droplets/circles
# returns the (x, y, radius in pixe, radius in um and estimated pL volume)
def get_Circles(img, fileName, showImg):
    "This finds the circles in a passed brightfield image and returns circle xy locations and radii (pixel, um and estimated volume in pL)"
    
    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    # min and max radius expected in pixles
    maxRadiusPixel = 100
    minRadiusPixel = 10
    
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = np.pi * minRadiusPixel**2
    params.maxArea = np.pi * maxRadiusPixel**2
    
    # look for light circles
    params.blobColor = 255;
    
    # Set Circularity filtering parameters- should be circle-like
    params.filterByCircularity = True
    params.minCircularity = 0.5
      
    # No Convexity filtering parameters
    params.filterByConvexity = False
    #.minConvexity = 0.2
    
    # No inertia filtering parameters
    params.filterByInertia = False
    
    #specifies that the circles cannot overlap I think, or that the centers of the circles
    # are a specific distance apart
    params.minDistBetweenBlobs = 0.7
      
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect circles
    keypoints = detector.detect(img)
        
    # Draw circles on image as red circles
    blank = np.zeros((1, 1)) 
    draw_circles = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    number_of_circles = len(keypoints)
    if number_of_circles == 0:
        #if no circles were detected then return
        print('No circles were detected in ' + fileName)
        all_Data = 0
        return all_Data
    
    # show/save or save image with circles
    if showImg == 1:
        make_CV2image(draw_circles,fileName, 1, 1)
        
    else:
        # still save the image 
        make_CV2image(draw_circles,fileName, 0, 1)
       
    #makes array to hold radii and grabs all the circle radii from keypoints
    circle_radii = np.empty([number_of_circles, 1])
    for i in range(number_of_circles):
        circle_radii[i, 0] = keypoints[i].size/2
    
    #makes array of [x, y] center coordinates 
    #******I don't know how to do this for the radii *********
    xyPoints = cv2.KeyPoint_convert(keypoints[:])
   
    #convert pixel radii to um and volume
    umRadii = circle_radii*2.2
    pLvolCircles = (umRadii**3)*((4/3)*np.pi) *0.001
   
    #combines data into matrix ( x center, y center, radius in pixels, radius in um, volume pL)
    all_Data = np.concatenate((xyPoints, circle_radii, umRadii, pLvolCircles), axis = 1)
    
    return all_Data

#callable function to threshold images for analysis
def image_Prep(img, channel, imageName, imageShow):
    
    channelName = channel + ' in '  + imageName
    if channel == 'BF':
        #increase brightness but stay  in 14 bit space
        image_rescale = exposure.rescale_intensity(img)
        #Dont' really need to threshold
        
        img_bw = cv2.convertScaleAbs(image_rescale, alpha =(255.0/65535.0))
        
        """
        if imageShow == 1:
            #plot and save modified image, needs to be converted to 8 bit to show
            #image8 = cv2.convertScaleAbs(img_bw, alpha =(255.0/65535.0))
            re = 600.0 / img_bw.shape[1]
            dim = (600, int(img_bw.shape[0] * re))
            resizeOutput = cv2.resize(img_bw, dim, interpolation=cv2.INTER_AREA)
        
            cv2.imshow(channelName, resizeOutput )
        
            #uncomment if you want to save these images
            #fileName =  channelName + '16bitRescale.png'
            #cv2.imwrite(fileName, resizeOutput)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        """
    else:
        # no modificaiton
        img_bw = img #/ (2**14)
        
    return img_bw


# callable function that returns the average intensity of ROIs
def get_ROI(fluorImg, circs, scaleFactor):
    '''returns a list of average pixel intensities of droplets in square ROI 
    calculated  from x,y, and radius passed to the function
    normFluorImg: should be the QD or fluorescent image (not normalized)
    that you want to get average intensity values for
    circs: a nparray of three columns (x, y, and r) that is returned from the get_Circles function
    scale Factor: how small the rectangle roi should be in the circle
    '''
    img_max = np.array(fluorImg.shape).max()
    img_min = 0
    # normalize image
    normFluorImg = fluorImg / (2**14 -1)
    # calculate the rectangles in the circular ROI  matrix is upper left x, upper left y, side of half of the square
    roiRect = [[int(circ[0]-circ[2]/math.sqrt(2)*scaleFactor), int(circ[1]-circ[2]/math.sqrt(2)*scaleFactor), int(circ[2]*math.sqrt(2)*scaleFactor)] for circ in circs]
    # remove any indicies outside the bounds of the image matrix
    roiRect = np.clip(roiRect, img_min, img_max)
    
    #grab average value in each roi
    avgRoiVal = [np.mean( normFluorImg[rect[1]:rect[1]+rect[2],rect[0]:rect[0]+rect[2]]) for rect in roiRect]

    #change from list to nparray
    avgIntensity = np.array(avgRoiVal)
    roiLoc = roiRect
    return roiLoc, avgIntensity
    

def multiple_dfs(df_list, imageNames, sheets, file_name, spaces):
    # converts a list of pd dataframes to an excel file
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter') 
    workbook=writer.book
    worksheet=workbook.add_worksheet(sheets)
    writer.sheets[sheets] = worksheet
    row = 0
    column = 0
    
    for i, dataframe in enumerate(df_list):
        # write name of image
        worksheet.write_string(row, column, imageNames[i] )
        
        # write number of total droplets found
        totalDroplets = dataframe.shape[0]
        worksheet.write_string(row+1, column , 'Number of total droplets')
        worksheet.write_string(row+1, column + 1, str(totalDroplets))
        
        
        if dataframe.columns.str.contains('Positive Droplets').any():
            worksheet.write_string(row + 2, column, 'Number of positive droplets')
            worksheet.write_string(row + 3, column, 'Fraction positive')
            colNames = [col for col in dataframe.columns if 'Positive Droplets'in col]
            
            for col in colNames:
                posDroplets = dataframe[col].sum()
                index = dataframe.columns.get_loc(col)
                worksheet.write_string(row + 2, column +  index+ 1,str(posDroplets ))
                worksheet.write_string(row + 3, column + index+ 1,str( posDroplets/totalDroplets))
            
        dataframe.to_excel(writer,sheet_name=sheets,startrow = row + 4 , startcol=column, header =True )   
        column = column + dataframe.shape[1] + spaces + 1
    
    writer.save()
    writer.close()

