#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:29:43 2018

@author: Jose M. Saavedra
This module contains the model specifications
"""
import tensorflow as tf      
from . import cnn_arch as arch
import os
  
def initializedModel(model_dir) :
    return os.path.exists(model_dir + '/init.init')    
        
#create a file indicating that init was done
def saveInitializationIndicator(model_dir) :
    with open(model_dir + '/init.init', 'w+') as f :
        f.write('1')    

#defining a model that feeds the Estimator
def model_fn (features, labels, mode, params):
    """The signature here is standard according to Estimators. 
       The output is an EstimatorSpec
    """
    #instance of the cnet
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else :
        is_training = False    
    if params['arch'] == 'MNIST':
        net = arch.mnistnet_fn(features, params['image_shape'], params['number_of_classes'], params['number_of_channels'], is_training)                        
    elif params['arch'] == 'SIMPLE_LARGER':
        net = arch.net2_fn(features, params['image_shape'], params['number_of_classes'], params['number_of_channels'], is_training)    
    elif params['arch'] == 'SKETCH':
        net = arch.sketchnet_fn(features, params['image_shape'], params['number_of_classes'], params['number_of_channels'], is_training)
    else :
        raise ValueError("network architecture is unknown")
    #    
    output = net["output"]
    #---------------------------------------
    idx_predicted_class = tf.argmax(output, 1)            
    predicted_probs = tf.nn.softmax(output, name="pred_probs")
    #--------------------------------------    
    # If prediction mode, predictions is returned
    predictions = { "predicted_probabilities": predicted_probs
                   }
    if mode == tf.estimator.ModeKeys.PREDICT:
        estim_specs = tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else : # TRAIN or EVAL
        #model initialization if params[ckpt] is defined. This is used for fine-tuning
        if mode == tf.estimator.ModeKeys.TRAIN and not initializedModel(params['model_dir']):             
            variables = tf.trainable_variables()
            print(variables)
            if 'ckpt' in params:
                if params['ckpt'] is not None:
                    print('---Loading checkpoint : ' + params['ckpt'])
                    """
                    assignment_map is very critical for fine-tunning
                    this must be a dictionary mapping
                    checkpoint_scope_name/variable_name : scope_name/variable
                    """
                    tf.train.init_from_checkpoint(ckpt_dir_or_file = params['ckpt'],                                                  
                                                  assignment_map = { v.name.split(':')[0]  :  v for v in variables })
                                                  
                    #save and indicator file
                    saveInitializationIndicator(params['model_dir'])
                    print('---Checkpoint : ' + params['ckpt'] + ' was loaded')
        #-----------------------------------------------------------------------
        idx_true_class = tf.argmax(labels, 1)            
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=idx_true_class, predictions=idx_predicted_class)    
        # Define loss - e.g. cross_entropy - mean(cross_entropy x batch)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = labels)
        loss = tf.reduce_mean(cross_entropy)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # to allow update [is_training variable] used by batch_normalization
        with tf.control_dependencies(update_ops) :
            optimizer = tf.train.AdamOptimizer(learning_rate= params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())        
        #EstimatorSpec 
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=idx_predicted_class,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})    
    
    return  estim_specs