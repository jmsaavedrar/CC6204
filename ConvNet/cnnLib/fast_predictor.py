'''
Created on Feb 21, 2019

@author:  jose.saavedra

A faster predictor loading a model previoously saved
'''
from tensorflow.contrib import predictor
import tensorflow as tf
from . import configuration as conf
from . import imgproc
from . import pmapping as pmap 
import numpy as np
from . import data
import os

class FastPredictor:

    def __init__(self, str_config, params):                                                    
        #reading configuration file    
        self.configuration = conf.ConfigurationFile(str_config, params['modelname'])
        self.modelname = self.configuration.getModelName()                
        self.processFun = imgproc.getProcessFun(self.configuration.getProcessFun())
        #snapShotDir must exist 
        assert os.path.exists(self.configuration.getSnapshotDir()), "Path {} does not exist".format(self.configuration.getSnapshotDir())
        #loading mean.dat and metadata.dat
        filename_mean = os.path.join(self.configuration.getDataDir(), "mean.dat")
        metadata_file = os.path.join(self.configuration.getDataDir(), "metadata.dat")
        #reading metadata
        self.image_shape = np.fromfile(metadata_file, dtype=np.int32)
        #         
        print("image shape: {}".format(self.image_shape))
        #load mean
        mean_img = np.fromfile(filename_mean, dtype=np.float32)
        self.mean_img = np.reshape(mean_img, self.image_shape.tolist())    
        #defining files for training and test    
        #loading model
        self.predictor = predictor.from_saved_model(os.path.join(self.configuration.getDataDir(),"cnn-model"))
        print("predictor loaded OK")
        mapping_file = os.path.join(self.configuration.getDataDir(), "mapping.txt")
        self.mapping = False            
        if  os.path.exists(mapping_file):
            self.class_mapping = pmap.PMapping(mapping_file)
            self.mapping = True
            
        self.device = params['device']
    def getDeepFeatures(self, filename):
        with tf.device(self.device):
            input_image =  data.input_fn_for_prediction(filename,
                                        self.image_shape,
                                        self.mean_img,
                                        self.configuration.getNumberOfChannels(),
                                        self.processFun)            
            predictions = self.predictor({"input":input_image})
            features = predictions['deep_features'][0]
            return features
                
    def predict(self, input_im):
        """
        input_im is a filename or a numpy array
        """
        with tf.device(self.device):
            input_image =  data.input_fn_for_prediction(input_im,
                                        self.image_shape,
                                        self.mean_img,
                                        self.configuration.getNumberOfChannels(),
                                        self.processFun)
            
            predictions = self.predictor({"input":input_image})
            predictions = predictions['predicted_probabilities'][0]
            if self.mapping :        
                names = [self.class_mapping.getClassName(item) for item in np.arange(len(predictions))]
            else :                  
                names = [item for item in np.arange(len(predictions))]
            idx_sorted = np.flip(np.argsort(predictions))
            sorted_names = [names[item] for item in idx_sorted]
            predictions = predictions[idx_sorted]
        return predictions, sorted_names
    
    def predict_on_list(self, list_of_images):
        with tf.device(self.device):
            input_image =  data.input_fn_for_prediction_on_list(list_of_images,
                                        self.image_shape,
                                        self.mean_img,
                                        self.configuration.getNumberOfChannels(),
                                        self.processFun)
            
            predictions = self.predictor({"input":input_image})
            predictions = predictions['predicted_probabilities']
            if self.mapping :
                names = [self.class_mapping.getClassName(item) for item in np.arange(predictions.shape[1])]
            else :
                names = [item for item in np.arange(predictions.shape[1])]
            idx_sorted = np.fliplr(np.argsort(predictions))
            sorted_predictions = []
            sorted_names = []        
            for idx, item_idx_sorted in enumerate(idx_sorted) :
                sorted_predictions.append(predictions[idx][item_idx_sorted])
                names_sorted = [names[item] for item in item_idx_sorted]
                sorted_names.append(names_sorted)
                    
        return np.array(sorted_predictions, dtype = np.float32), sorted_names