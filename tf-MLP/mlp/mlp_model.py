'''
Created on Mar 4, 2019

@author: jsaavedr
'''

import tensorflow as tf

def gaussian_weights(shape,  mean, stddev):
    return tf.truncated_normal(shape, 
                               mean = mean, 
                               stddev = stddev)


def fc_layer(_input, size, name, use_sigmoid=True):
    """
    a fully connected layer
    """         
    # shape is a 1D tensor with 4 values
    num_features_in = _input.get_shape().as_list()[-1]
    #reshape to  1D vector    
    shape = [num_features_in, size]
    W = tf.Variable(gaussian_weights(shape, 0.0, 0.02), name=name)     
    b = tf.Variable(tf.zeros(size))
    #just a  multiplication between input[N_in x D]xW[N_in x N_out]
    layer = tf.add( tf.matmul(_input, W) ,  b)        
    if use_sigmoid:
        layer=tf.nn.sigmoid(layer)
    return  layer

def mlp_fn(features,  input_size, n_classes):    
    with tf.variable_scope("mlp_scope"):                                
        # fc 1
        features = tf.reshape(features, [-1, input_size])
        print(" features {} ".format(features.get_shape().as_list()))
        fc1 = fc_layer(features, 100, name = 'fc1')    
        #fc5 = layers.dropout_layer(fc5, 0.8)
        print(" fc1: {} ".format(fc1.get_shape().as_list()))
        #fc 6
        fc2 = fc_layer(fc1, 100, name = 'fc2')            
        print(" fc2: {} ".format(fc2.get_shape().as_list()))
        #fully connected
        fc3 = fc_layer(fc2, n_classes, name = 'fc3', use_sigmoid = False )
        print(" fc3: {} ".format(fc3.get_shape().as_list()))
                        
    return {"output": fc3}

#defining a model that feeds the Estimator
def model_fn (features, labels, mode, params):
    """The signature here is standard according to Estimators. 
       The output is an EstimatorSpec
    """
    
    net = mlp_fn(features, params['input_size'], params['number_of_classes'])                    
    #    
    output = net["output"]
                
    predicted_probs = tf.nn.softmax(output, name="predicted_probs")
    #--------------------------------------    
    # If prediction mode, predictions is returned
    predictions = { "predicted_probs": predicted_probs,                    
                   }
    if mode == tf.estimator.ModeKeys.PREDICT:
        estim_specs = tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else : # TRAIN or EVAL        
        idx_true_class = tf.argmax(labels, 1)
        idx_predicted_class = tf.argmax(output,1)            
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=idx_true_class, predictions=idx_predicted_class)    
        # Define loss - e.g. cross_entropy - mean(cross_entropy x batch)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = labels)                 
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())        
        #EstimatorSpec 
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=idx_predicted_class,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})    
    
    return  estim_specs







