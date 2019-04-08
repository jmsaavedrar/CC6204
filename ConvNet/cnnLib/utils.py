#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:46:48 2018

@author: jsaavedr

"""
import numpy as np

#to uint8
def toUINT8(image) :
    if image.dtype == np.float64 :
        image = image * 255
    elif image.dtype == np.uint16 :
        image = image >> 8        
    image[image<0]=0
    image[image>255]=255
    image = image.astype(np.uint8, copy=False)
    return image