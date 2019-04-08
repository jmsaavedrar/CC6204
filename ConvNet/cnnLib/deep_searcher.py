#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:05:34 2018

@author: jsaavedr
"""
#DeepSearcher
from . import fast_predictor as fp
from . import norm
import numpy as np
import struct
import os
import time
import enum


class JNorm(enum.Enum):
    SQUARE_ROOT_UNIT = 1
    UNIT = 2
    

class DeepSearcher : 
    
    def __init__(self, str_config, params):
        #loading predictor
        self.fast_predictor = fp.FastPredictor(str_config, params)
        self.data = []
        self.names = []
        self.dist = []
        self.dim = -1 
        self.size = -1
        self.norm = JNorm.SQUARE_ROOT_UNIT
    
    def load(self, path):
        t_start = time.time()
        feature_file = os.path.join(path, "features.des")
        name_file = os.path.join(path, "names.txt")        
        #open binary file
        f_feature = open(feature_file, "r+b")
        header = f_feature.read(3*4) #reading 3 int32, eachone with 4 bytes
        header = np.array(struct.unpack("i"*3, header))
        self.size = header[0]
        self.dim = header[1]
        #reading data
        data_size = self.size * self.dim # dim times number of objects
        data = f_feature.read(data_size * 4)
        data = np.array(struct.unpack("f"*data_size, data))
        #reshape in header[0] x header[1]
        self.data = data.reshape([header[0], header[1]])                
        f_feature.close()
        
        #reading names
        with open(name_file) as f_name:
            self.names = [line.rstrip() for line in f_name]
        
        if self.norm == JNorm.SQUARE_ROOT_UNIT :
            self.data = norm.squareRootNorm(self.data)
        print("Data was loaded OK")
        elapsed = (time.time() - t_start ) * 1000
        print(">>> shape of data {}".format(self.data.shape))        
        print(">>> loaded took {} ms".format(elapsed))        
        
    def getFeature(self, input_image):
        """
        input_image is a filename or a numpy array
        """
        #assert(os.path.exists(filename))
        t_start = time.time()        
        features = self.fast_predictor.getDeepFeatures(input_image)        
        if self.norm == JNorm.SQUARE_ROOT_UNIT :
            deep_features = norm.squareRootNorm(features)
        elapsed = (time.time() - t_start ) * 1000
        print(">>> getFeature took {} ms".format(elapsed))
        return deep_features
    
    def search(self, fvec, k):        
        t_start = time.time()
        dist = (self.data - fvec) ** 2
        dist = np.sum(dist, axis = 1)
        self.dist = np.sqrt(dist)                                
        sorted_idx = sorted (range(self.size), key = lambda x : self.dist[x])        
        elapsed = (time.time() - t_start ) * 1000
        print(">>> search took {} ms".format(elapsed))
        if k == -1 :
            return sorted_idx,  dist[sorted_idx]
        else :
            return sorted_idx[0:k], dist[sorted_idx[0:k]]
                
    def getName(self, idx):
        """return tne name of the object with id = idx"""
        return self.names[idx]
    
    def getDist(self, idx):
        """return tne dist of the object with id = idx"""
        return self.dist[idx]