#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:02:08 2018

@author: jose.saavedra

Implementation of layers
"""
import tensorflow as tf
import numpy as np

#gaussian weights 
def gaussian_weights(shape,  mean, stddev):
    return tf.truncated_normal(shape, 
                               mean = mean, 
                               stddev = stddev)

#xavier weights TODO    
def xavier_weights(shape):
    return 0

#convolution layer using stride = 1, is_training is added to set BN approproiately    
def conv_layer(input, shape, name, stride = 1, is_training = False):    
    #weights are initialized according to a gaussian distribution
    W =  tf.Variable(gaussian_weights(shape, 0.0, 0.01), name=name)     
    #weights for bias ares fixed as constants 0
    b = tf.Variable(tf.zeros(shape[3]), name='bias_'+name)
    return tf.nn.relu(
            tf.layers.batch_normalization(
                tf.add(tf.nn.conv2d(
                        input, 
                        W, 
                        strides=[1, stride, stride, 1], 
                        padding='SAME'), b), scale = True, training = is_training))

#pooling layer that uses max_pool, a square  kernel is used
def max_pool_layer(input, kernel, stride):
    return tf.nn.max_pool(input,  
                          [1, kernel, kernel, 1], 
                          [1, stride, stride, 1], 
                          padding = 'SAME' )

#pooling layer that uses avg_pool, a square kernel is used
def avg_pool_layer(input, kernel, stride):
    return tf.nn.avg_pool(input,  
                          [1, kernel, kernel, 1], 
                          [1, stride, stride, 1], 
                          padding = 'SAME' )
#global average pooling  GAP
def gap_layer(input):
    kernel_h = input.get_shape().as_list()[1]
    kernel_w = input.get_shape().as_list()[2]
    return tf.nn.avg_pool(input,  [1, kernel_h, kernel_w, 1], [1, 1, 1, 1], padding = 'VALID' )

#fully-connected layer fc
def fc_layer(input, size, name, use_relu=True): 
    layer_shape_in =  input.get_shape()
    # shape is a 1D tensor with 4 values
    num_features_in = layer_shape_in[1:4].num_elements()
    #reshape to  1D vector
    input_reshaped = tf.reshape(input, [-1, num_features_in])
    shape = [num_features_in, size]
    W = tf.Variable(gaussian_weights(shape, 0.0, 0.02), name=name)     
    b = tf.Variable(tf.zeros(size))
    #just a  multiplication between input[N_in x D]xW[N_in x N_out]
    layer = tf.add( tf.matmul(input_reshaped, W) ,  b)        
    if use_relu:
        layer=tf.nn.relu(layer)
    return  layer

#dropout
def dropout_layer(input, prob):
    """prob is a float value, representing the probability that each element is kept"""
    return tf.nn.dropout(input, prob)


    

