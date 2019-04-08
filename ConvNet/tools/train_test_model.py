#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:30:08 2018
@author: jose.saavedra

A convolutional neural network tool 
This implementation are based on using the classes Estimator and Dataset
For more details see cnnLib.cnn

"""

import argparse
import sys
import os
import params as _params
sys.path.insert(0,_params.cnn_lib_location)
import cnnLib.cnn  as cnn
import cnnLib.configuration as conf    
import cnnLib.pmapping as pmap
import numpy as np

#-----------main----------------------
if __name__ == '__main__':                        
    parser = argparse.ArgumentParser(description = "training / testing x models")
    parser.add_argument("-mode", type=str, choices=['test', 'train', 'predict', 'save'], help=" test | train ", required = True)
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required = False)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)
    parser.add_argument("-config", type=str, help=" <optional>, a configuration file", required = False)
    parser.add_argument("-ckpt", type=str, help=" <optional>, it defines the checkpoint for training<fine tuning> or testing", required = False)        
    parser.add_argument("-image", type=str, help=" <optional>, a filename for an image to be tested in -predict mode-", required = False)        
    
    pargs = parser.parse_args() 
    if pargs.config != None :
        configuration_file = pargs.config  
    else :
        configuration_file = _params.configuration_file
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)    
    configuration.show()
    #it is also possible to define the id of the device
    if pargs.device is None :
        device_name = "/cpu:0"
    else :
        device_name = "/" + pargs.device + ":0"
    params = {'device': device_name,
              'modelname' : pargs.name}    
    if pargs.ckpt is not None:
        params['ckpt'] = pargs.ckpt        
    my_cnn = cnn.CNN(configuration_file, params)      
    run_mode = pargs.mode
    if run_mode == 'train' :        
        my_cnn.train()
    elif run_mode == 'test' :
        my_cnn.test()        
    elif run_mode == 'predict' :
        print(pargs.image)
        assert os.path.exists(pargs.image), "-image is required"
        prediction = my_cnn.predict(pargs.image)[0]        
        probs = prediction['predicted_probabilities']                        
        idx_class = np.argmax(probs)
        prob = probs[idx_class]
        print("Class: {} [ {} ] ".format(idx_class, prob))
        mapping_file = os.path.join(configuration.getDataDir(), "mapping.txt")            
        if  os.path.exists(mapping_file):
            class_mapping = pmap.PMapping(mapping_file)
            print("Predicted class [{}]".format(class_mapping.getClassName(idx_class)))
        else :
            print("Predicted class [{}]".format(idx_class))
    elif run_mode == 'save' :
        my_cnn.save_model()    
    print("OK   ")   
    
    