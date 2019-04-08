#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on march, 27
@author: jsaavedr
"""
import argparse
import sys
import os
import params as _params
import numpy as np                 
sys.path.insert(0, _params.cnn_lib_location)    
import cnnLib.deep_searcher  as dsearcher
import cnnLib.imgproc  as imgproc

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description = "Evaluation")    
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
    #loading test dataset [test.txt]    
    test_file = os.path.join(pargs.dir, "test.txt")
    assert os.path.exists(test_file), "file for test [{}] does not exist".format(test_file)
    with open(test_file) as t_file :            
        lines = [line.rstrip() for line in t_file]             
    lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ] 
    filenames, labels = zip(*lines_)
    labels = np.array([int(label) for label in labels])
    #load labels 
    lfile = os.path.join(pargs.dir, "labels.npy")
    dlabels = np.fromfile(lfile, dtype = np.int32)
    #loading dseracher 
    dsearch = dsearcher.DeepSearcher(configuration_file, params)
    dsearch.load(pargs.dir)
    mAP = 0                      
    for i, item in enumerate(filenames) :        
        print(item)       
        im_query = imgproc.readSketch(item, 256, 236)
        fvec = dsearch.getFeature(im_query)           
        idx, _ = dsearch.search(fvec, -1)        
        rel = dlabels[idx]
        
        correct_position = np.where(rel == labels[i])[0]        
        id_rel = np.arange(0,len(correct_position),1) +1        
        pr = id_rel / (correct_position +1 ) 
        apr = np.mean(pr)
        mAP =  mAP + apr
        print(apr) 
        if (i+1) % 10 == 0:
            print("Progress {} %".format(100*(i+1)/len(filenames)))
    mAP  = mAP / len(filenames) 
    print("mAP {}".format(mAP))    
    
        