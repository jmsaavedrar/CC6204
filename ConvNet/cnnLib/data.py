#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 2018
@author: jsaavedr

Description: A list of function to create tfrecords
for regular images
"""
#import cv2
import os
import sys
import numpy as np
import random
import tensorflow as tf
from . import utils
import skimage.io as io
import skimage.color as color

#%% int64 should be used for integer numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#%% byte should be used for string  | char data
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#%% float should be used for floating point data
def _float_feature(value):    
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#%%
#%%a set of input functions
#%%
#define the input function for training and testing, the data is read from tfrecords
def input_fn(filename, image_shape, mean_img, is_training, configuration):     
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(lambda x: parser_tfrecord(x, 
                                                     image_shape, 
                                                     mean_img, 
                                                     configuration.getNumberOfClasses(), 
                                                     configuration.getNumberOfChannels()))
    dataset = dataset.batch(configuration.getBatchSize())   
    if is_training:
        #shuffle the batches
        dataset = dataset.shuffle(configuration.getNumberOfBatches())
        dataset = dataset.repeat(configuration.getNumberOfEpochs())
        # for testing shuffle and repeat are not required    
    return dataset
#%%
#define the input function for prediction
def input_fn_for_prediction(the_image, image_shape, mean_img, number_of_channels, processFun):
    """
    the_image: it can be a filename or a numpy array
    preocessFun is the function to be used for processing the input image
    """        
    if isinstance(the_image, str) : 
        image = readImage(the_image, number_of_channels)    
    else :
        image = loadImage(the_image, number_of_channels)        
    image = processFun(image, (image_shape[0], image_shape[1]))    
    image = np.reshape(image, image_shape)
    image = image - mean_img
    image = image.astype(np.float32)
    image = np.reshape(image, [1, image_shape[0], image_shape[1], number_of_channels])
    return image     
#%%
#define the input function for prediction
def input_fn_for_prediction_on_list(list_of_images, image_shape, mean_img, number_of_channels, processFun):    
    batch_of_images = np.zeros([len(list_of_images), image_shape[0], image_shape[1], image_shape[2]])
    for idx, the_image in enumerate(list_of_images) : 
        if isinstance(the_image, str) :
            image = readImage(the_image, number_of_channels)    
        else:
            image = loadImage(the_image, number_of_channels)
            
        image = processFun(image, (image_shape[0], image_shape[1]))                
        image = np.reshape(image, image_shape)
        batch_of_images[idx, : , : , :] = image - mean_img
    batch_of_images = batch_of_images.astype(np.float32)    
    return batch_of_images
#%% 
def loadImage(np_image, number_of_channels):
    if len(np_image.shape) == 2:
        number_of_channels_in = 1
    else :
        number_of_channels_in = np_image.shape[2]        
    assert(number_of_channels_in in [1,3])
    image = np_image
    if (number_of_channels != number_of_channels_in):
        if number_of_channels  == 1 :
            image = color.rgb2gray(image)          
            assert(len(image.shape) == 2)
        if number_of_channels  == 3 :
            image = color.gray2rgb(image)            
            assert(len(image.shape) == 3)
    image = utils.toUINT8(image)        
    return image
#%%
def readImage(filename, number_of_channels):
    """ readImage using skimage """    
    if number_of_channels  == 1 :
        #image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)            
        image = io.imread(filename, as_grey = True)
        image = utils.toUINT8(image)        
        assert(len(image.shape) == 2)
    elif number_of_channels == 3 :
        #image = cv2.imread(filename)
        image = io.imread(filename)
        if(len(image.shape) == 2) :
            image = color.gray2rgb(image)
        image = utils.toUINT8(image)        
        assert(len(image.shape) == 3)
    else:
        raise ValueError("number_of_channels must be 1 or 3")
    if not os.path.exists(filename):
        raise ValueError(filename + " does not exist!")
    return image
#%% 
def validateLabels(labels) :
    """Process labels checking if these are in the correct format [int]
       labels need to be integers, from 0 no NCLASSES -1 
    """  
    new_labels = [int(label) for label in labels]
    label_set = set(new_labels)
    #checking the completness of the label set
    if (len(label_set) == max(label_set) + 1) and (min(label_set) == 0):
        return new_labels
    else:            
        raise ValueError("Some codes are missed in label set! {}".format(label_set))
        
#%%
def readDataFromTextFile(str_path, dataset = "train" , shuf = True):    
    """read data from text files
    and apply shuffle by default 
    """            
    datafile = os.path.join(str_path, dataset + ".txt")    
    assert os.path.exists(datafile)        
    # reading data from files, line by line
    with open(datafile) as file :        
        lines = [line.rstrip() for line in file]     
        if shuf:
            random.shuffle(lines)
        lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ] 
        filenames, labels = zip(*lines_)
        labels = validateLabels(labels)    
        #labels = [int(label) for label in labels]
    return filenames, labels

#%%
"""creating tfrecords
Important!! Number of classes is no longer computed, it will be a parameter in configuration
"""    
def createTFRecordFromList(filenames, labels, target_shape, processFun, tfr_filename):
    h = target_shape[0]
    w = target_shape[1]
    number_of_channels = target_shape[2]
    writer = tf.python_io.TFRecordWriter(tfr_filename)    
    assert len(filenames) == len(labels)
    if number_of_channels == 1  :
        mean_image = np.zeros([h,w], dtype=np.float32)    
    else:
        mean_image = np.zeros([h,w,number_of_channels], dtype=np.float32)    
    for i in range(len(filenames)):        
        if i % 500 == 0 or (i + 1) == len(filenames):
            print("---{}".format(i))           
        image = readImage(filenames[i], number_of_channels)
        image = processFun(image, (h, w))        
        #create a feature                
        feature = {'train/label': _int64_feature(labels[i]), 
                   'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}
        #crate an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))        
        #serialize to string an write on the file
        writer.write(example.SerializeToString())
        mean_image = mean_image + image / len(filenames)            
    #serialize mean_image
    writer.close()
    sys.stdout.flush()
    return mean_image
#%%

def createTFRecord(str_path, id_type, im_size, number_of_channels, processFun):    
    """ 
    id_type = 0: only train 
              1: only test    
              2: both
    im_shape: size of the input image (height, width)          
    number_of_channels: this is the number of channels of the input
    processFun: processing function which depends on the problem we are trying to solve
    """
    #saving metadata
    image_shape = np.array([im_size[0], im_size[1], number_of_channels])    
    #------------- creating train data
    if ( id_type + 1 ) & 1 : # train   ( 0 + 1 ) & 1  == 1 
        filenames, labels = readDataFromTextFile(str_path, dataset="train", shuf = True)    
        tfr_filename = os.path.join(str_path, "train.tfrecords")
        training_mean = createTFRecordFromList(filenames, labels, image_shape, processFun, tfr_filename)
        print("train_record saved at {}.".format(tfr_filename))
        #saving training mean
        mean_file = os.path.join(str_path, "mean.dat")
        print("mean_file {}".format(training_mean.shape))
        training_mean.astype(np.float32).tofile(mean_file)
        print("mean_file saved at {}.".format(mean_file))  
    #-------------- creating test data    
    if ( id_type + 1 ) & 2 : # test ( 1 + 1 ) & 2  == 2
        filenames, labels = readDataFromTextFile(str_path, dataset="test", shuf = True)  
        tfr_filename = os.path.join(str_path, "test.tfrecords")
        createTFRecordFromList(filenames, labels, image_shape, processFun, tfr_filename)
        print("test_record saved at {}.".format(tfr_filename))    
            
    metadata_array = image_shape
    #saving metadata file    
    metadata_file = os.path.join(str_path, "metadata.dat")
    metadata_array.astype(np.int32).tofile(metadata_file)
    print("metadata_file saved at {}.".format(metadata_file))      
# %% parser sk
#---------parser_tfrecord for mnist
def parser_tfrecord(serialized_example, im_size, mean_img, number_of_classes, number_of_channels):    
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/image': tf.FixedLenFeature([], tf.string),
                                        'train/label': tf.FixedLenFeature([], tf.int64)
                                        })
    image = tf.decode_raw(features['train/image'], tf.uint8)    
    image = tf.reshape(image, [im_size[0], im_size[1], number_of_channels])
    image = tf.cast(image, tf.float32) - tf.cast(tf.constant(mean_img), tf.float32)
    #image = image * 1.0 / 255.0    
    #one-hot 
    label = tf.one_hot(tf.cast(features['train/label'], tf.int32), number_of_classes)
    label = tf.reshape(label, [number_of_classes])
    label = tf.cast(label, tf.float32)
    return image, label
