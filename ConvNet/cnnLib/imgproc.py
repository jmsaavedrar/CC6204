#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:32:07 2018

This modulo contains the function in charge of pre-processing the input
It should be the same used during training, testing and prediction
@author: jsaavedr
"""
import skimage.transform as transf
from . import utils
import skimage.morphology as morph
import skimage.io as io
import numpy as np

#%%    
def processImage(image, imsize):
    """
    imsize = (h,w)
    """
    #resize uses format (w,h)
    image_out = transf.resize(image, imsize)    
    image_out = utils.toUINT8(image_out)
    return image_out

#%%    
def processLAR(image, imsize):
    """
    imsize = (h,w)
    """
    #resize uses format (w,h)
    image_out = transf.resize(image, imsize)    
    image_out = utils.toUINT8(image_out)
    return image_out


def processMnistImage(image, imsize):    
    """
    imsize = (h,w)
    """
    #resize uses format (w,h)
    image_out = transf.resize(image, imsize)
    image_out = utils.toUINT8(image_out)
    return image_out

#return as certain function for preprocessing the inputs 
def getProcessFun(str_name = 'default') :
    if str_name == 'default' :
        return processImage
    else :
        raise ValueError("{} is not a valid processFun name".format(str_name))

def normalizeSketch(mat_sketch, im_size, obj_size, _erosion = True ):
    
    """
    mat_sketch is a numpy array: uint8 [0-255]
    This function produces a normalized sketch image
    im_size : size of the output image im_size x im_size
    obj_size: size of the maximun side of the sketch
    """    
    assert len(mat_sketch.shape) == 2, "input is not a 1-channel image"    
    assert obj_size < im_size , "{} should be < {} ".format(obj_size, im_size)
    bin_image = np.ones(mat_sketch.shape, np.uint8)
    th = 128
    bin_image[mat_sketch < th] = 0
    #get stroke wider       
    if _erosion : 
        bin_image = morph.erosion(bin_image, morph.square(3))            
    rows, cols = np.where(bin_image == 0)
    target_im = np.ones((im_size, im_size), np.uint8)*255
    if len(rows) + len(cols) > 0 :
        min_y = np.min(rows)
        max_y = np.max(rows)
        min_x = np.min(cols)
        max_x = np.max(cols)    
        #extractign sketch region    
        im_sketch = bin_image[min_y: (max_y +1), min_x: (max_x + 1)]    
        max_size_sk = np.max(im_sketch.shape)    
        factor = obj_size / max_size_sk
        #rescaling sketch image  
        target_sk = utils.toUINT8(transf.rescale(im_sketch*255, factor))
        target_sk[:,0]=255
        target_sk[0,:]=255    
        target_sk[:,-1]=255
        target_sk[-1,:]=255
        sk_height=target_sk.shape[0]
        sk_width=target_sk.shape[1]
        offset_x = np.int((im_size - sk_width)*0.5) 
        offset_y = np.int((im_size - sk_height)*0.5)    
        target_im[offset_y: offset_y + sk_height, offset_x : offset_x + sk_width]=target_sk
    return target_im
    
def readSketch(filename, im_size, obj_size, for_data = False):
    """
    filename is the path to the image
    im_size is the size of the target image
    obj_size is the size of the maximum size of the sketch (foreground)
    """
    im_query = utils.toUINT8(io.imread(filename, as_gray = True))
    erosion = True
    if for_data :
        erosion = False
    im_query = normalizeSketch(im_query, im_size, obj_size, erosion)
    return im_query

def readSketchData(filename):
    """
    filename is the path to the image
    im_size is the size of the target image
    obj_size is the size of the maximum size of the sketch (foreground)
    """
    im_query = utils.toUINT8(io.imread(filename, as_gray = True))
    bin_image = np.ones(im_query.shape, np.uint8)
    th = 128
    bin_image[im_query < th] = 0
    return bin_image * 255
