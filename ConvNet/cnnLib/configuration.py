#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:00:28 2018

@author: jsaavedr
"""
from configparser import SafeConfigParser

class ConfigurationFile:
    """
     An instance of ConfigurationFile contains required parameters to train a 
     convolutional neural network
    """    
    def __init__(self, str_config, modelname):
        config = SafeConfigParser()
        config.read(str_config)
        self.sections = config.sections()                
        if modelname in self.sections:
            try :
                self.modelname = modelname                
                self.arch = config.get(modelname, "ARCH")
                self.process_fun = 'default'
                if 'PROCESS_FUN' in config[modelname] is not None :
                    self.process_fun = config[modelname]['PROCESS_FUN']                                                    
                self.number_of_classes = config.getint(modelname,"NUM_CLASSES")
                self.number_of_iterations= config.getint(modelname,"NUM_ITERATIONS")
                self.dataset_size = config.getint(modelname, "DATASET_SIZE")
                self.test_size = config.getint(modelname, "TEST_SIZE")
                self.batch_size = config.getint(modelname, "BATCH_SIZE")
                self.estimated_number_of_batches =  int ( float(self.dataset_size) / float(self.batch_size) )
                self.estimated_number_of_batches_test = int ( float(self.test_size) / float(self.batch_size) )
                #snapshot time sets when temporal weights are saved (in steps)
                self.snapshot_time = config.getint(modelname, "SNAPSHOT_TIME")
                #test time sets when test is run (in seconds)
                self.test_time = config.getint(modelname, "TEST_TIME")
                self.lr = config.getfloat(modelname, "LEARNING_RATE")
                #snapshot folder, where training data will be saved
                self.snapshot_prefix = config.get(modelname, "SNAPSHOT_DIR")
                #number of estimated epochs
                self.number_of_epochs = int ( float(self.number_of_iterations) / float(self.estimated_number_of_batches) )
                #folder where tf data is saved. Used for training and testing
                self.data_dir = config.get(modelname,"DATA_DIR")
                self.channels = config.getint(modelname,"CHANNELS")                
                
                assert(self.channels == 1 or self.channels == 3)                
            except Exception:
                raise ValueError("something wrong with configuration file " + str_config)
        else:
            raise ValueError(" {} is not a valid section".format(modelname))
        
    def getModelName(self):
        return self.modelname
    
    def getArch(self) :
        return self.arch
    
    def getProcessFun(self):
        return self.process_fun
       
    def getNumberOfClasses(self) :
        return self.number_of_classes
    
    def getNumberOfIterations(self):
        return self.number_of_iterations
    
    def getNumberOfEpochs(self):
        return self.number_of_epochs
    
    def getDatasetSize(self):
        return self.dataset_size
    
    def getBatchSize(self):
        return self.batch_size
    
    def getNumberOfBatches(self):
        return self.estimated_number_of_batches
    
    def getNumberOfBatchesForTest(self):
        return self.estimated_number_of_batches_test
    
    def getSnapshotTime(self):
        return self.snapshot_time
    
    def getTestTime(self):
        return self.test_time
    
    def getSnapshotDir(self):
        return self.snapshot_prefix
    
    def getNumberOfChannels(self):
        return self.channels
    
    def getDataDir(self):
        return self.data_dir
    
    def getLearningRate(self):
        return self.lr    
    
    def isAValidSection(self, str_section):
        return str_section in self.sections
    
    def show(self):
        print("ARCH: {}".format(self.getArch()))
        print("NUM_ITERATIONS: {}".format(self.getNumberOfIterations()))
        print("DATASET_SIZE: {}".format(self.getDatasetSize()))
        print("LEARNING_RATE: {}".format(self.getLearningRate()))
        print("NUMBER_OF_BATCHES: {}".format(self.getNumberOfBatches()))
        print("NUMBER OF EPOCHS: {}".format(self.number_of_epochs))
        print("SNAPSHOT_DIR: {}".format(self.getSnapshotDir()))
        print("DATA_DIR: {}".format(self.getDataDir()))