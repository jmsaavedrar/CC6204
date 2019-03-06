"""
This is an implementation to test tfrecords
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def parser_tfrecord_simple(serialized_example, im_size):
    """
    A simple parser for deserializing the input
    """    
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/feature': tf.FixedLenFeature([36], tf.float32),
                                        'train/label': tf.FixedLenFeature([], tf.int64)
                                        })
    
    feature = features['train/feature']
    label = features['train/label']
    label = tf.cast(label, tf.float32)        
    return feature, label

#%% MAIN
if __name__ == '__main__':        
    datadir = "/home/vision/smb-datasets/SBIR/Quickdraw-10c"
          
    filename = os.path.join(datadir, 'test.tfrecords' ) 
    assert(os.path.exists(filename))
    mean_file = os.path.join(datadir, "mean.dat")
    metadata_file = os.path.join(datadir, "metadata.dat")
    assert(os.path.exists(mean_file))
    assert(os.path.exists(metadata_file))
    BATCH_SIZE = 10    
    #reading metadata file to know the image_shape and number of classes 
    image_shape = np.fromfile(metadata_file, dtype=np.int32)
    print("image_shape: {}".format(image_shape))        
    mean_vector =np.fromfile(mean_file, dtype=np.float32)    
    print("mean vector {}".format(mean_vector))
    #---------------read TFRecords data  for training
    data_train = tf.data.TFRecordDataset(filename)    
    data_train = data_train.map(lambda x: parser_tfrecord_simple(x, image_shape))    
    data_train = data_train.batch(BATCH_SIZE)    
    iterator = tf.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)
    next_batch = iterator.get_next()
    #tensor that initialize the iterator:
    training_init_op = iterator.make_initializer(data_train)    
    
    with tf.Session() as sess:        
        sess.run(training_init_op)                     
        fig, xs = plt.subplots(1, BATCH_SIZE)        
        while True:
            try:
                features, labels = sess.run(next_batch)
                for i in range(BATCH_SIZE):                                                                
                    print("{} {}".format(features[i], labels[i]))
            except IndexError:
                break 
            except tf.errors.OutOfRangeError:
                break