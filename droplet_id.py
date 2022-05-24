# -*- coding: utf-8 -*-
__author__ = 'Lael Wentland'
__copyright__ = 'Copyright 2021, GH labs'
__credits__ = ['Lael Wentlad', 'Louise Hansen', 'Josh Bishop']
__license__ = 'Apache 2.0'
__version__ = '2.0'
__maintainer__ = 'Josh Bishop'
__email__ = 'josh.bishop@ghlabs.org'
__status__ = 'Development'
__fullname__ = 'Polydisperse Droplet Identification'

"""
This is Python code that processes .TIF images of droplets. The image must
contain a brightfield subimage and one or more fluorescence subimages.
The code detects the droplets in the brightfield slice using an OpenCV blob
(circle) detector. Then the code gathers the average fluorescence intensity
in each of the fluorescence subimages from rectangular ROIs circumscribed by the
detected circles.

Users need to
1) specify the folder location of the .TIF images,
2) name the subimages, and
3) choose whether to save analyzed images.

Updated 20211129 by Josh Bishop
"""

#import os          # to help move you in the correct directory
import cv2
import numpy as np
import pandas as pd
import glob
from skimage import exposure
from tifffile import imread # to read stacked tiff file
from PIL import Image, ImageFilter

# import the following extra files which should be in the same folder as MAIN
from inference import lam_dvv

#------Change the constants to apply script to a specific image set ----------
#DIRECTORY_NAME = r'C:/Users/JoshuaBishop/Global Health Labs, Inc/Polydroplets on Lumira - Documents/General/Data/WP2 - Hardware/Luis/20220510 - Chlymidia - first try'
DIRECTORY_NAME = r'C:/temp'
SUBIMAGE_NAMES = ['BF', 'FAM'] # list in order the image acquisition mode
SUBIMAGE_THRESHOLDS = {'FAM': 0.83, 'Cy5': 0.6}
CIRCLE_SQUARE_SCALE_FACTOR = 0.9 # scale square ROI inscribed in detected droplet circle to avoid measuring edge intensities
PIXEL_MICRON_SCALE_FACTOR = 2.2  # scale camera pixels to microns
SAVE_IMAGE_OPTION = True   # False = no don't save, True = yes please save images!
# path to Image folder - copy absolute path of the folder
#replicate_number = 5
#negatives_number = 5 # just in case you have different number of negative images than regular replicates
#directory_name = r'C:/Users/JoshuaBishop/GH+ Labs/Digital Assay Feasibility - Documents/General/Data/Interns/Lael W/20210826 Louise MultiModal Data/20210825Images'
#image_slice_names = ['Cy5', 'FAM', 'BF'] # list in order the image acquisition mode
# choose threshold manually if you want, if threshold_values dictionary is left blank
# then the computer will automatically grab the last few images and generate a threshold
# based on those images, under the assumption that they are negatives
#------------------------------------

def main():
    writer = pd.ExcelWriter(f'{DIRECTORY_NAME}/droplet_id.xlsx')
    print(f'Save location: {DIRECTORY_NAME}/droplet_id.xlsx')
    if SAVE_IMAGE_OPTION:
        annotated_images = []
    for image_data in [ImageData(image_name) for image_name in list_dir_images()]:
        print(f'Saving to sheet: {image_data.image_name[0:30]}')
        image_data.df.to_excel(writer, sheet_name=image_data.image_name[0:30])
        if SAVE_IMAGE_OPTION:
            annotated_images += image_data.annotated_images
    writer.save()
    if SAVE_IMAGE_OPTION:
        #print(annotated_images[0])
        image_list = [Image.fromarray(image) for image in annotated_images]
        image_list[0].save(f'{DIRECTORY_NAME}/droplet_id.pdf', "PDF", resolution=300.0, save_all=True, append_images=image_list[1:-1])

class ImageData():
    """A class to store droplet id image data during processing of a stacked TIFF image from (e.g., Nikon T2) inverted fluoresence microscope"""
    def __init__( self
                , file_name
                , subimage_names = SUBIMAGE_NAMES
                , subimage_thresholds = SUBIMAGE_THRESHOLDS
                , circle_square_scale_factor = CIRCLE_SQUARE_SCALE_FACTOR
                , pixel_micron_scale_factor = PIXEL_MICRON_SCALE_FACTOR):

        self.file_name = file_name
        if 'BF' not in subimage_names:
            raise Exception(f'[ERROR] Exactly one subimage in the list {subimage_names} must be named \'BF\'')
        self.image_name = file_name[file_name.rindex('\\')+1:file_name.find('.tif')]
        self.subimage_names = subimage_names
        self.brightfield_index = subimage_names.index('BF')
        self.fluorescence_indices = [i for i, _ in enumerate(subimage_names) if i != self.brightfield_index]
        self.subimage_thresholds = subimage_thresholds
        threshold_missing = [name for name in subimage_names if name != 'BF' and name not in [*subimage_thresholds]]
        if len(threshold_missing) != 0:
            print('[INFO] Positive droplet threshold' + ('' if len(threshold_missing) == 1 else 's') + f' not supplied for {str(threshold_missing)}')
        self.circle_square_scale_factor = circle_square_scale_factor
        self.pixel_micron_scale_factor = pixel_micron_scale_factor
        self.keypoints = []
        self.df = pd.DataFrame(columns = [ 'circle_center_x'
                                         , 'circle_center_y'
                                         , 'circle_pixel_radius'
                                         , 'droplet_micron_radius'
                                         , 'droplet_picoliter_volume'
                                         , 'square_upperleft_x'
                                         , 'square_upperleft_y'
                                         , 'square_pixel_length'
                                         ])
        self.annotated_images = [None] * len(subimage_names)
        print(f'Processing image: {self.image_name}')
        self.get_brightfield_features()
        self.get_fluorescence_intensities()

    def get_total_circles(self):
        return self.df.shape[0]

    # function to threshold BF for analysis
    def get_brightfield_features(self):
        image = imread(self.file_name)
        if np.shape(image)[-1] != len(self.subimage_names):
            raise Exception(f'[ERROR] Image has a different number of subimages ({np.shape(image)[-1]}) than provided names ({self.subimage_names})')
        image = image[:, :, self.brightfield_index]                             # load BF image data
        image = exposure.rescale_intensity(image)                               # brighten image
        image = cv2.convertScaleAbs(image, alpha =(255.0/65535.0))              # change image to 8 bit
        image = Image.fromarray(image.astype('uint8'))                          # convert image to matrix from PIL
        image = image.filter(ImageFilter.UnsharpMask(radius=10, percent=200))   # filter image
        erosion_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        image = cv2.erode(np.asarray(image),erosion_se)                         # convert image back to matrix and enhance with erosion
        image = cv2.dilate(image, erosion_se)                                   # enhance with dilation

        # Set our filtering parameters
        # Initialize parameter settiing using cv2.SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        maxRadiusPixel = 120                        # in pixels
        minRadiusPixel = 6                          # in pixels
        params.filterByArea = True                  # looking for circles in area range
        params.minArea = np.pi * minRadiusPixel**2
        params.maxArea = np.pi * maxRadiusPixel**2
        params.blobColor = 255;                     # looking for light circles
        params.filterByCircularity = True           # looking for circles
        params.minCircularity = 0.5
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minDistBetweenBlobs = 0.7
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect circles
        self.keypoints = detector.detect(image)
        # print(self.keypoints[0])

        # show/save or save image with circle annotations
        if SAVE_IMAGE_OPTION:
            self.annotated_images[self.brightfield_index] = self.annotate_image(image)

        if len(self.keypoints) == 0:
            # if no circles were detected then return
            print(f'[INFO] No circles were detected in {self.file_name}')
            return False

        # stores data as (x center, y center, radius in pixels, radius in µm, volume pL)
        self.df[['circle_center_x', 'circle_center_y', 'circle_pixel_radius', 'droplet_micron_radius', 'droplet_picoliter_volume']] = pd.DataFrame(
                [[ keypoint.pt[0]
                , keypoint.pt[1]
                , keypoint.size/2
                , self.pixel_micron_scale_factor*keypoint.size/2
                , (self.pixel_micron_scale_factor*keypoint.size/2**3)*((4/3)*np.pi)*0.001]
                for keypoint
                in self.keypoints])

        image_max = np.array(image.shape).max()
        image_min = 0
        # calculate the squares inscribed in the detected droplet circles
        # matrix is upper left x, upper left y, side of half of the square
        self.df[['square_upperleft_x', 'square_upperleft_y', 'square_pixel_length']] = np.clip(
            [ list(map( lambda var: int(var*self.circle_square_scale_factor)
                    , [x-r/np.sqrt(2), y-r/np.sqrt(2), r*np.sqrt(2)]))
                for x, y, r
                in self.df[['circle_center_x', 'circle_center_y', 'circle_pixel_radius']].values
            ]
            , image_min
            , image_max)

        return True

    # function that returns the average intensity of ROIs
    def get_fluorescence_intensities(self):
        '''
        returns a list of average pixel intensities of droplets in square ROI
        calculated from x, y, and radius passed to the function
        image_data:
        scale Factor: how small the rectangle roi should be in the circle compared to the diameter
        '''
        for i in self.fluorescence_indices:                       # for each FL image
            image = imread(self.file_name)[:,:,i] # load image data
            image = image / (2**14-1)                      # microscope camera is 14-bit, normalize for easier thresholding in the range [0, 1]
            name = self.subimage_names[i]

            #grab mean intensity value in image inside each square
            self.df[name+'_square_mean_intensity'] = [
                    np.mean(image[y:y+n, x:x+n])
                    for x, y, n
                    in self.df[['square_upperleft_x', 'square_upperleft_y', 'square_pixel_length']].astype(int).values #are double brackets required?
                ]
            if SAVE_IMAGE_OPTION:
                self.annotated_images[i] = self.annotate_image(image)
            
        if name in [*self.subimage_thresholds]:
            self.df[name+'_droplet_positivity'] = self.df[name+'_square_mean_intensity'] > self.subimage_thresholds[name]

        return True

    # function to shorten the production and saving of CV2 images
    def annotate_image(self, image, w = 1200):
        re = float(w) / image.shape[1]
        dim = (w, int(image.shape[0] * re))
        image = image.astype(np.uint8)
        #image = cv2.drawKeypoints(image, self.keypoints, np.zeros((1, 1)), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for kp in self.keypoints:
            cv2.circle(image, (int(kp.pt[0]),int(kp.pt[1])), color=(255,0,0), radius=int(kp.size/2), thickness=3)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        return image

def list_dir_images(directory = DIRECTORY_NAME):
    #os.chdir(directory)            # save generated images in specified directory
    path = directory + '/*.tif'    # look for all .tif images in specified directory
    file_names = glob.glob(path)
    file_names.sort()
    return file_names

if __name__ == "__main__":
    main()

