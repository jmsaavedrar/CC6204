#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:31:01 2018

@author: jsaavedr
"""

#visual search

import argparse
import sys
import cv2
import params as _params
import numpy as np  
import skimage.transform as trans
import skimage.morphology as morph               
sys.path.insert(0, _params.cnn_lib_location)    
import cnnLib.deep_searcher  as dsearcher
import cnnLib.utils  as utils
import cnnLib.imgproc  as imgproc
import skimage.io as io
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Deep Searcher")    
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)        
    parser.add_argument("-dir", type=str, help=" path where data is saved", required = True)    
    parser.add_argument("-config", type=str, help=" <optional> configuration file", required = True)    
    pargs = parser.parse_args() 
    configuration_file = pargs.config
    device_name = "/" + pargs.device + ":0"
    params = {'device': device_name, 
              'modelname' : pargs.name
              }       
        
    dsearch = dsearcher.DeepSearcher(configuration_file, params)
    dsearch.load(pargs.dir)
    #resulting image
    tsize= 128
    result_image = np.zeros((tsize*2,tsize*6,3), np.uint8)    
        
    while True :
        filename = input("Image: ")
        #reading a sketch query        
        im_query1 = imgproc.readSketch(filename, 256, 236)
        im_query = cv2.resize(im_query1, (tsize,tsize))
        im_query = cv2.cvtColor(im_query, cv2.COLOR_GRAY2BGR)                
        result_image[0:tsize, 0:tsize, :] = im_query
        fvec = dsearch.getFeature(im_query1)        
        idx, dist = dsearch.search(fvec, 10)
        for i in range(10):
            im_result=dsearch.getName(idx[i])            
            print("{}\t{}\n".format(dsearch.getName(idx[i]), dsearch.getDist(idx[i])))
            image = cv2.imread(im_result, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (tsize,tsize))
            row = i // 5
            col = (i % 5)  + 1
            result_image[row*tsize:(row+1)*tsize, col*tsize:(col+1)*tsize,:] = image
            
        cv2.imshow("result", result_image)
        cv2.waitKey()
