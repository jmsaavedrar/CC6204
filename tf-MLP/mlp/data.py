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
    dataset = dataset.map(lambda x: parser_tfrecord(x,input_params['number_of_classes'], input_params['input_size'], mean_vector))
    dataset = dataset.batch(input_params['batch_size'])   
    if is_training:
        #shuffle the batches
        dataset = dataset.shuffle(input_params['number_of_batches'])
        dataset = dataset.repeat(input_params['number_of_epochs'])
        # for testing shuffle and repeat are not required    
    return dataset
#%%
def input_fn_for_prediction(filename, mean_vector, input_size):                
    image = readImage(filename)        
    features = cvision.getHOG(image)
    features = features - mean_vector
    features = features.astype(np.float32)
    features = np.reshape(features, [1, input_size])
    return features
 
def readImage(filename):
    """ readImage using skimage """    
    image = io.imread(filename, as_gray = True)    
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
        raise ValueError("Some codes are missed in the label set! {}".format(label_set))
        
#%%
def readDataFromTextFile(str_path, dataset = "train" , shuf = True):    
    """read data from text files
    and  shuffle it by default
    str_path: path where data can be found
    dataset: train or test
    shuf: Truel or Flase for shuffling 
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
def createTFRecordFromList(filenames, labels, input_size, tfr_filename):    
    """
    filenames: a list of image filenames
    labels: a list of target labels
    input_size: feature vector size
    trf_filename: filename of the destine tfrecord 
    """
    writer = tf.python_io.TFRecordWriter(tfr_filename)    
    assert len(filenames) == len(labels), "filenames and labels should have the same size"
    mean_vector = np.zeros(input_size, dtype=np.float32)    
    for i in range(len(filenames)):        
        if i % 100 == 0 or (i + 1) == len(filenames):
            print("---{}".format(i))           
        image = readImage(filenames[i])
        #here  the feature vector is computed
        features = cvision.getHOG(image)
        assert features.size == input_size, "{} != {}".format(features.size, input_size)            
        #create a feature                
        feature = {'train/feature': _floatArray_feature(features), 
                   'train/label': _int64_feature(labels[i])}
        #crate an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))        
        #serialize to string and write in file
        writer.write(example.SerializeToString())
        mean_vector = mean_vector + features / len(filenames)            
    writer.close()
    sys.stdout.flush()
    return mean_vector
#%%

def createTFRecord(str_path, id_type, input_size):    
    """
    A function for creating a tfrecords for training and test 
    str_path: path where data is stored
    id_type = 0: only train 
              1: only test    
              2: both
    input_size: feature vector size                  
    """        
    #------------- creating train data
    if ( id_type + 1 ) & 1 : # train   ( 0 + 1 ) & 1  == 1 
        filenames, labels = readDataFromTextFile(str_path, dataset="train", shuf = True)    
        tfr_filename = os.path.join(str_path, "train.tfrecords")
        training_mean_fv = createTFRecordFromList(filenames, labels, input_size, tfr_filename)
        print("train_record saved at {}.".format(tfr_filename))
        #saving training mean feature vector
        mean_file = os.path.join(str_path, "mean.dat")        
        training_mean_fv.astype(np.float32).tofile(mean_file)
        print("mean_file saved at {}.".format(mean_file))  
    #-------------- creating test data    
    if ( id_type + 1 ) & 2 : # test ( 1 + 1 ) & 2  == 2
        filenames, labels = readDataFromTextFile(str_path, dataset="test", shuf = True)  
        tfr_filename = os.path.join(str_path, "test.tfrecords")
        createTFRecordFromList(filenames, labels, input_size, tfr_filename)
        print("test_record saved at {}.".format(tfr_filename))                    
    #saving metadata file
    metadata = np.array([input_size])    
    metadata_file = os.path.join(str_path, "metadata.dat")
    metadata.astype(np.int32).tofile(metadata_file)
    print("metadata_file saved at {}.".format(metadata_file))      
# %% parser sk
#---------parser_tfrecord for mnist
def parser_tfrecord(serialized_example,   number_of_classes, input_size, mean_vector ):    
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/feature': tf.FixedLenFeature([input_size], tf.float32),
                                        'train/label': tf.FixedLenFeature([], tf.int64)
                                        })    
    feature = features['train/feature']    
    feature = feature - tf.cast(tf.constant(mean_vector), tf.float32)
    #one-hot 
    label = tf.one_hot(tf.cast(features['train/label'], tf.int32), number_of_classes)
    label = tf.reshape(label, [number_of_classes])
    label = tf.cast(label, tf.float32)
    return feature, label
