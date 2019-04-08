#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:10:35 2018

@author: jsaavedr
This programa read and show the data saved as tfrecords
"""

import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def parser_tfrecord_simple(serialized_example, im_size, number_of_channels):    
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/image': tf.FixedLenFeature([], tf.string),
                                        'train/label': tf.FixedLenFeature([], tf.int64)
                                        })
    image = tf.decode_raw(features['train/image'], tf.uint8)    
    if number_of_channels == 1 :
        image = tf.reshape(image, [im_size[0], im_size[1]])    
    else :
        image = tf.reshape(image, [im_size[0], im_size[1], number_of_channels])         
    label = tf.cast(features['train/label'], tf.int32)        
    return image, label

if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description = "test data saved as tfrecord")    
    parser.add_argument("-pathname", type = str, help = "<string> path where the data is stored" , required = True)
    parser.add_argument("-type", type = int, help = "<int> 0: train, 1: test", required = True )
    pargs = parser.parse_args()           
    str_path = pargs.pathname      
    filename = os.path.join(str_path, 'train.tfrecords' if pargs.type == 0 else 'test.tfrecords' ) 
    assert(os.path.exists(filename))
    mean_file = os.path.join(str_path, "mean.dat")
    metadata_file = os.path.join(str_path, "metadata.dat")
    assert(os.path.exists(mean_file))
    assert(os.path.exists(metadata_file))
    BATCH_SIZE = 10    
    #reading metadata file to know the image_shape and number of classes 
    image_shape = np.fromfile(metadata_file, dtype=np.int32)
    print("image_shape: {}".format(image_shape))
    number_of_channels = image_shape[2]
    print(image_shape)
    mean_img =np.fromfile(mean_file, dtype=np.float32)    
    mean_img = np.reshape(mean_img, image_shape.tolist())            
    #---------------read TFRecords data  for training
    data_train = tf.data.TFRecordDataset(filename)    
    data_train = data_train.map(lambda x: parser_tfrecord_simple(x, image_shape, image_shape[2]))    
    data_train = data_train.batch(10)    
    iterator = tf.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)
    next_batch = iterator.get_next()
    #tensor that initialize the iterator:
    training_init_op = iterator.make_initializer(data_train)    
    
    with tf.Session() as sess:        
        sess.run(training_init_op)                     
        fig, xs = plt.subplots(1, BATCH_SIZE)        
        while True:
            try:
                img, label = sess.run(next_batch)                                                                
                for i in range(BATCH_SIZE) :                                        
                    if number_of_channels == 1 : 
                        xs[i].imshow(img[i,:,:], cmap = 'gray')
                    elif number_of_channels == 3 :
                        xs[i].imshow(img[i,:,:, :])
                    else :
                        raise ValueError(" number of channels exceeds 3, images can't be displayed!!")
                    xs[i].axis('off')
                    xs[i].set_title(label[i][0])
                    #cv2.waitKey()
                plt.waitforbuttonpress()
            except IndexError:
                break 
            except tf.errors.OutOfRangeError:
                break