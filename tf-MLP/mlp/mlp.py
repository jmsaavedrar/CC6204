
from . import mlp_model as model
from . import data as data
import tensorflow as tf
import numpy as np
import os

         
class MLP:
    def __init__(self, params):                         
        #reading configuration file            
        self.device = params['device']
        self.modeldir = params['model_dir']         
        self.datadir = params['data_dir']        
        self.learning_rate = params['learning_rate']
        self.number_of_classes = params['number_of_classes']
        self.number_of_iterations = params['number_of_iterations']
        self.batch_size = params['batch_size']
        self.data_size = params['data_size']
        self.number_of_batches = np.round(self.data_size / self.batch_size) 
        self.number_of_epochs = np.round(self.number_of_iterations / self.number_of_batches)
        
        #loading mean and metadata
        filename_mean = os.path.join(self.datadir, "mean.dat")
        metadata_file = os.path.join(self.datadir, "metadata.dat")
        #reading metadata
        self.input_size = np.fromfile(metadata_file, dtype=np.int32)[0]        
        #load mean
        self.mean_vector = np.fromfile(filename_mean, dtype=np.float32)
        print("mean_vector {}".format(self.mean_vector.shape))                
        #defining files for training and test
        self.filename_train = os.path.join(self.datadir, "train.tfrecords")
        self.filename_test = os.path.join(self.datadir, "test.tfrecords")         
        #print(" mean {}".format(self.mean_img.shape))        
        self.input_params = { 'batch_size' : self.batch_size,
                            'number_of_batches': self.number_of_batches,
                            'number_of_epochs': self.number_of_epochs,
                            'input_size': self.input_size,
                            'number_of_classes' : self.number_of_classes,
                }
        
        self.estimator_params = {'learning_rate' : self.learning_rate,
                             'number_of_classes' : self.number_of_classes,                                                                                                                    
                             'model_dir': self.modeldir,
                             'input_size': self.input_size}
    def train(self):
        """training"""
        #-using device gpu or cpu
        with tf.device(self.device):
            estimator_config = tf.estimator.RunConfig( model_dir = self.modeldir,
                                                       save_checkpoints_steps=1000,
                                                       keep_checkpoint_max=10)
            
            classifier = tf.estimator.Estimator( model_fn = model.model_fn,
                                                 config = estimator_config,
                                                 params = self.estimator_params )
            #
            tf.logging.set_verbosity(tf.logging.INFO) # Just to have some logs to display for demonstration
            #training
            
            train_spec = tf.estimator.TrainSpec(input_fn = lambda: data.input_fn(self.filename_train, self.input_params, self.mean_vector, True),
                                                 max_steps = self.number_of_iterations)
            #max_steps is not useful when inherited checkspoint is used
            eval_spec = tf.estimator.EvalSpec(input_fn = lambda: data.input_fn(self.filename_test, self.input_params, self.mean_vector, False),
                                              start_delay_secs = 30)
            #
            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
            
    def test(self):
        """test checkpoint exist """
        assert os.path.exists(os.path.join(self.modeldir, "checkpoint")), "Checkpoint file does not exist in {}".format(self.modeldir)
        """testing"""
        with tf.device(self.device):
            tf.logging.set_verbosity(tf.logging.INFO)
            estimator_config = tf.estimator.RunConfig(model_dir = self.modeldir)
            
            classifier = tf.estimator.Estimator( model_fn = model.model_fn,
                                                 config = estimator_config,
                                                 params = self.estimator_params )
             
            result = classifier.evaluate(input_fn = lambda: data.input_fn(self.filename_test, self.input_params, self.mean_vector, True))
            print(result)        
            
            