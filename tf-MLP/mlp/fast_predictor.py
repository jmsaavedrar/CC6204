'''
Created on Feb 21, 2019

@author:  jose.saavedra

A faster predictor loading a model previoously saved
'''
from tensorflow.contrib import predictor
import tensorflow as tf 
import numpy as np
from . import data
import os

class FastPredictor:

    def __init__(self, params):                                                    
        self.device = params['device']
        self.modeldir = params['model_dir']         
        self.datadir = params['data_dir']                
        self.number_of_classes = params['number_of_classes']                
        #loading mean and metadata
        filename_mean = os.path.join(self.datadir, "mean.dat")
        metadata_file = os.path.join(self.datadir, "metadata.dat")
        #reading metadata
        self.input_size = np.fromfile(metadata_file, dtype=np.int32)[0]        
        #load mean
        self.mean_vector = np.fromfile(filename_mean, dtype=np.float32)                                                     
    
        #loading model
        self.predictor = predictor.from_saved_model(os.path.join(self.datadir,"mlp-model"))
        print("predictor loaded OK")        
                    
    def predict(self, filename):
        with tf.device(self.device):
            input_image =  data.input_fn_for_prediction(filename,                                        
                                        self.mean_vector, self.input_size)            
            predictions = self.predictor({"input":input_image})
            predictions = predictions['predicted_probs'][0]            
            idx_sorted = np.flip(np.argsort(predictions), 0)                        
        return idx_sorted[0], predictions[idx_sorted][0]