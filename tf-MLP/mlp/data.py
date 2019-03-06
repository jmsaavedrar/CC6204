'''
Created on Mar 5, 2019

@author: jsaavedr
'''

import os
import sys
import numpy as np
import random
import tensorflow as tf
import skimage.io as io
import skimage.transform as transf
import skimage.color as color
from . import  cvision
import cv2

#%% int64 should be used for integer numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#%% byte should be used for string  | char data
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#%% float should be used for floating point data
def _float_feature(value):    
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#%% float array
def _floatArray_feature(value):    
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
#%%
#%%a set of input functions
#%%
#define the input function for training and testing, the data is read from tfrecords
def input_fn(filename,  input_params, mean_vector, is_training):     
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(lambda x: parser_tfrecord(x,10, input_params['K'], mean_vector))
    dataset = dataset.batch(input_params['batch_size'])   
    if is_training:
        #shuffle the batches
        dataset = dataset.shuffle(input_params['number_of_batches'])
        dataset = dataset.repeat(input_params['number_of_epochs'])
        # for testing shuffle and repeat are not required    
    return dataset
#%%
 

def readImage(filename):
    """ readImage using skimage """    
    image = io.imread(filename, as_gray = True)    
    image = cvision.toUINT8(image)                 
    return image

def processImage(image, shape):
    image = transf.resize(image, shape)
    image = cvision.toUINT8(image)    
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
def createTFRecordFromList(filenames, labels, target_shape, tfr_filename):    
    writer = tf.python_io.TFRecordWriter(tfr_filename)    
    assert len(filenames) == len(labels)
    mean_vector = np.zeros(72, dtype=np.float32)    
    for i in range(len(filenames)):        
        if i % 100 == 0 or (i + 1) == len(filenames):
            print("---{}".format(i))           
        image = readImage(filenames[i])
        image = processImage(image, target_shape)
        features = cvision.getHistogramOfOrientations(image, 72)            
        #create a feature                
        feature = {'train/feature': _floatArray_feature(features), 
                   'train/label': _int64_feature(labels[i])}
        #crate an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))        
        #serialize to string an write on the file
        writer.write(example.SerializeToString())
        mean_vector = mean_vector + features / len(filenames)            
    #serialize mean_image
    writer.close()
    sys.stdout.flush()
    return mean_vector
#%%

def createTFRecord(str_path, id_type, im_size):    
    """ 
    id_type = 0: only train 
              1: only test    
              2: both
    im_shape: size of the input image (height, width)          
    number_of_channels: this is the number of channels of the input
    processFun: processing function which depends on the problem we are trying to solve
    """
    #saving metadata
    image_shape = np.array(im_size)    
    #------------- creating train data
    if ( id_type + 1 ) & 1 : # train   ( 0 + 1 ) & 1  == 1 
        filenames, labels = readDataFromTextFile(str_path, dataset="train", shuf = True)    
        tfr_filename = os.path.join(str_path, "train.tfrecords")
        training_mean = createTFRecordFromList(filenames, labels, image_shape, tfr_filename)
        print("train_record saved at {}.".format(tfr_filename))
        #saving training mean
        mean_file = os.path.join(str_path, "mean.dat")
        print("mean_file {} {}".format(training_mean.shape, training_mean))
        training_mean.astype(np.float32).tofile(mean_file)
        print("mean_file saved at {}.".format(mean_file))  
    #-------------- creating test data    
    if ( id_type + 1 ) & 2 : # test ( 1 + 1 ) & 2  == 2
        filenames, labels = readDataFromTextFile(str_path, dataset="test", shuf = True)  
        tfr_filename = os.path.join(str_path, "test.tfrecords")
        createTFRecordFromList(filenames, labels, image_shape, tfr_filename)
        print("test_record saved at {}.".format(tfr_filename))    
            
    metadata_array = image_shape
    #saving metadata file    
    metadata_file = os.path.join(str_path, "metadata.dat")
    metadata_array.astype(np.int32).tofile(metadata_file)
    print("metadata_file saved at {}.".format(metadata_file))      
# %% parser sk
#---------parser_tfrecord for mnist
def parser_tfrecord(serialized_example,   number_of_classes, k, mean_vector ):    
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/feature': tf.FixedLenFeature([k], tf.float32),
                                        'train/label': tf.FixedLenFeature([], tf.int64)
                                        })    
    feature = features['train/feature']    
    feature = feature - tf.cast(tf.constant(mean_vector), tf.float32)
    #one-hot 
    label = tf.one_hot(tf.cast(features['train/label'], tf.int32), number_of_classes)
    label = tf.reshape(label, [number_of_classes])
    label = tf.cast(label, tf.float32)
    return feature, label
