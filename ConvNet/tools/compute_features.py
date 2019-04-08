#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:17:59 2018

@author: jsaavedr
Compute deep descriptor from a list of images
"""
import argparse
import sys
import os
import math
import time
import numpy as np
import params as _params
sys.path.insert(0, _params.cnn_lib_location)
import cnnLib.configuration as conf    
import cnnLib.fast_predictor as fp
import cnnLib.imgproc as imgproc

if __name__ == '__main__':
    #the following path may be changed if necessary                 
    
    parser = argparse.ArgumentParser(description = "computer deep features for a list of images")    
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)    
    parser.add_argument("-list", type=str, help=" <optional>, a list of images", required = False)    
    parser.add_argument("-batch_size", type=int, help=" <optional>, size of the batch used for prediction", required = False)    
    parser.add_argument("-odir", type=str, help=" path where data is saved", required = False)    
    parser.add_argument("-config", type=str, help=" <optional> configuration file", required = True)        
    pargs = parser.parse_args()
    configuration_file = pargs.config        
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)
    device_name = "/" + pargs.device + ":0"
    params = {'device': device_name,
              'modelname' : pargs.name}           
    if pargs.batch_size is None :
        batch_size = 50
    else :
        batch_size = pargs.batch_size
            
    faster_predictor=fp.FastPredictor(pargs.config, params)
    
    filename = pargs.list
    assert os.path.exists(filename), "file {} does not exist".format(filename)
    
    #line syntax  filename\tclassname
    with open(filename) as file:
        lines = [line.rstrip() for line in file]     
    lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ] 
    filenames, labels = zip(*lines_)        
    #assert list_of_images
    assert len(filenames) > 0, "list of images is empty"
    #compute the size of each batch
    n_batches = math.ceil(len(lines) / batch_size)    
    feature_file = os.path.join(pargs.odir, "features.des")
    name_file = os.path.join(pargs.odir, "names.txt")
    label_file = os.path.join(pargs.odir, "labels.npy")
    f_feature = open(feature_file, "w+b")
    f_name = open(name_file, "w+t")    
    #Compute the size of the deep_learning layer, this will be the size of the deep feature vector                    
    features = faster_predictor.getDeepFeatures(filenames[0])
    deep_vector_size = features.size
    print("Size of deep feature vector: {}".format(deep_vector_size))
    #store 3-int32 header: number_of_images, sise_of_deep_feature_vector, batch_size
    header = np.array([len(lines), deep_vector_size, batch_size], dtype=np.int32)
    f_feature.write(np.int32(header))
    #saving features in batch
    it = 0
    avg_elapsed = 0
    for item in filenames:        
        t_start = time.time()
        #list_of_images = lines[i*batch_size : min(len(lines), (i+1)*batch_size)]        
        deep_features = faster_predictor.getDeepFeatures(item)
        #deep_features = faster_predictor.getDeepFeatures(im_query)             
        f_feature.write(deep_features.astype(np.float32))                        
        elapsed = (time.time() - t_start)*1000
        avg_elapsed = avg_elapsed + elapsed
        it = it + 1
        if (it % 1000) == 0 :
            print("Progress: {}% ".format( int( 10000 * (it + 1) /len(filenames) ) / 100 ))
            print(" deep feature -> {} ".format(deep_features.shape))
            print("Average time per query {}".format(avg_elapsed / 1000))
            avg_elapsed = 0
                        
    #saving names
    for line in filenames :
        f_name.write(line + "\n")
    
    array_labels = np.array(labels, dtype = np.int32)
    array_labels.tofile(label_file)
    print("{} features saved at {} ".format(len(filenames), feature_file))    
    print("{} names saved at {} ".format(len(filenames), name_file))
    print("{} labels saved at {} ".format(len(filenames), label_file))
        
    f_feature.close()
    f_name.close()
#         
