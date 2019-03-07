'''
Created on Mar 5, 2019

@author: jsaavedr

A set of functions that applies computer vision methods
'''

import skimage.feature as feat
import skimage.transform as trans
import skimage.io as io
from scipy import ndimage
import numpy as np
import sys
import cv2

def toUINT8(image) :
    if image.dtype == np.float64 :
        image = image * 255
    elif image.dtype == np.uint16 :
        image = image >> 8        
    image[image<0]=0
    image[image>255]=255
    image = image.astype(np.uint8, copy=False)
    return image

def getHOG(image):
    image = trans.resize(image, (64,64))
    image = toUINT8(image)    
    fd= feat.hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector = True)    
    return fd
    
def getHistogramOfOrientations(image, k):
    """
    image : numpy image
    k: number of buckets
    """    
    sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype = np.float32)
    sobel_y = np.transpose(sobel_x)
    gx = ndimage.convolve(image.astype(np.float32), sobel_x, mode = 'constant', cval = 0)
    gy = ndimage.convolve(image.astype(np.float32), sobel_y, mode = 'constant', cval = 0)
    ang = np.arctan2(gy, gx)
    ang[ang < 0] = ang[ang < 0] + np.pi    
    magnitude = np.sqrt(np.power(gx, 2.0) + np.power(gy, 2.0) )
    magnitude[ magnitude < 0.1] = 0    
    indx = np.round((ang / np.pi) * (k - 1))    
    histogram = np.zeros(shape = k, dtype = np.float32)    
    for i in range(k) :
        val = np.sum(magnitude[indx == i])        
        histogram[i] = val            
    norm2 = np.linalg.norm(histogram, 2)
    histogram = histogram / norm2    
    return histogram


if __name__ == '__main__' : 
    filename = sys.argv[1]
    image = io.imread(filename, as_gray = True)
    image = toUINT8(image)    
    #h = getHistogramOfOrientations(image, 36)
    h = getHOG(image)
    print(h)
    print(h.shape)
        
    
    

    
    

