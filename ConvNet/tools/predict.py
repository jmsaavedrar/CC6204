'''
Created on Feb 20, 2019

@author:  jose.saavedra

A simple program for prediction using a saved model 
'''
import argparse
import params as _params
import sys
sys.path.insert(0,_params.cnn_lib_location)
import cnnLib.fast_predictor as fp
import time
    
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description = "training / testing x models")    
    parser.add_argument("-config", type=str, help=" configuration file", required = True)       
    parser.add_argument("-name", type=str, help=" name of the model", required = True)
    parser.add_argument("-list", type = str, help = "a list of  images for prediction", required = False)     
    parser.add_argument("-image", type = str, help = "an image", required = False)
    parser.add_argument("-device", type = str, choices = ['gpu', 'cpu'], help = "device gpu or cpu", required = False)
    
    
    pargs = parser.parse_args()
    str_device = "/cpu:0"
    if not pargs.device is None :
        str_device = "/" + pargs.device + ":0"
        
    params = {'device': str_device, 
              'modelname' : pargs.name}
                    
    faster_predictor=fp.FastPredictor(pargs.config, params)
    if not pargs.list is None :
        with open(pargs.list) as f_in :
            list_images = [item.strip() for item in f_in]            
        n_images = len(list_images)
        avg = 0    
        for item in list_images :
            start = time.time()                
            predicted_probs, predicted_classes = faster_predictor.predict(item)
            print("{} -> {}".format(predicted_classes[0], predicted_probs[0]))        
            print("OK")
            end = time.time()
            avg = avg + end - start    
        print("Average elapsed time {} ".format(avg / n_images))
    
    elif not pargs.image is None :
        filename = pargs.image
        while True : 
            start = time.time()                
            predicted_probs, predicted_classes = faster_predictor.predict(filename)
            print("{} -> {}".format(predicted_classes[0], predicted_probs[0]))                
            end = time.time()
            print("Elased time {} ".format(end - start))            
            filename = input("Image: ")
            while len(filename) == 0 :
                filename = input("Image: ")