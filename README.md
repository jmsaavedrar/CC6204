# ConvNet Description 
ConvNet is a set of modules based on TensorFlow aiming to make the process of creating, training and testing convolutional neural models easier. A module for a fast prediction, after saving the trained model, is also included.

# Structure
## ccnLib: A package containing the core modules
   - data.py: A set of functions to create and read tfrecords. This also includes input functions for estimators.
   - imgproc.py: A set of functions for image processing. The user can also add customized functions.
   - layers.py: A set of function for creating neural layers.
   - cnn_arch.py: A set of convolutional neural architectures. Here, customized nets are defined.
   - cnn_model.py: Here, the model is created which defines the optimizer and the specifications for training and testing.
   - configuration.py: Here the class ConfigurationFile is defined, it reads hyper-parameters from a configuration file.
   - cnn.py: Here, the class CNN is implemented. A detailed description of this class is discussed forward.
   - fast_predictor.py: This contains the class *FastPredictor*, that allows us to run predictions in an efficient way (this avoids reloading the model in each prediction). This class requires saving the checkpoints (a saved model), which can be obtained using the method -save- of CNN class.
## tools: A set of tools for creating tf_records and for training, testing and predicting using CNNs. 
   **- create_data.py:** This generates *tf_records*  from a data folder containing two files: *train.txt* and *test.txt* each one specifing the images for training an testing, respectively. The text files should be formatted in a two column style with the following syntax \<image path>\t\<label>. The label columns should be in a 0-indexed format. You can use the tool *processInputFile.py* to convert string labels into 0-indexed integers.
      * Parameters
         - type: [int, 0: only train, 1: only test, 2: both]
         - imwheight: height of the target image
         - imwidth: width of the target image
         - config: path to the the configuration file [See Configuration Section ](#the-configuration-file)
         - name: name of section in configuration file. The name is very important since the configuration file may include multiple configuration sections.
         
   After creating data, the following files are also created:
        - mean.dat storing the mean of the training images 
        - metadata.data storing the shape of the images [H,W,CH].
        
  **Note**: Before creating data, we recommend to read the [Preparing Data Section](#preparing-data).
  
  Example
  
  > python3.6 tools/create_data.py -type 2 -imheight [height] -imwidth [width] -config [config-file] -name [name-model]
  
   **- train_test_model.py** [train, test, predict or save a cnn model]
      * Parameters
         - mode: [train | test | predit | save ]
         - device: [cpu | gpu]
         - ckpt: It defines a checkpoint for training or testing. In case of training this will be used for fine-tuning.
         - image: A filename used only in *predict* mode. Predict model is deprecated, prefer fast-prediction.
         - config: A configuration file with the required hyper-parameters
         - name: Name of the section using in the configuration file
   
# The class CNN
An CNN object is equiped with the following member functions:
   - train
   - test
   - predict(image) [deprecated]
   - predict_on_list(list_of_images) [deprecated]
   - save: To save a model for future prediction (it is recommended for faster predictions)
   
To instatiate a CNN object a *configuration file* together with the following parameteres are required. 
   - 'name' : name of the model that is the name of the section in configuration file
   - 'device' : It must be 'gpu' or 'cpu'

These parameters are passed through a dictionary. The parameters that must be defined in the configuration file are described below.

# The Configuration File
   - ARCH [name of the cnn architecture]. This name is used in cnn_model.py
   - NUM_ITERATIONS = [int]
   - NUM_CLASSES = [int]
   - DATASET_SIZE = [int, number of images for training]
   - TEST_SIZE = [int, numnber of images for testing]
   - BATCH_SIZE =  [int]
   - SNAPSHOT_TIME = [int, save a snapshot each this many iterations]
   - TEST_TIME =  [int, start evaluating after waiting for this many seconds.]
   - LEARNING_RATE = [int]
   - SNAPSHOT_DIR = [path where the partial models are saved. This is the default path for saving the model]
   - DATA_DIR = [path where data feed must be found, data feed are train.txt and test.txt]
   - CHANNELS = [int]

# How to train a model
Use train_test_mode.py using -mode test 
# How to test a model
Use train_test_mode.py using -mode train

# Preparing Data
1. Format the input file to have integer labels. This will produce *.jfile and *.mapping 
   > python3.6 tools/processInputFile.py -path [data path]
   
   *- path* is the path where data can be found. It needs to contain a list.txt file wiht all the images to process. The list file should be formatted in a two-column style indicating the *name of the image* together with the *class name* (separated by a tab). The class name could be a string, the program will check the data and convert it into a valid format *(\*.jfile)*, where the class names are all  integers and zero-indexed. The mapping between the real name an the indexed names will be saved in the  *\*.mapping file.*      
   
2. Divide file into test.txt and train.txt
   > python3.6 tools/divideFile.py  -path [data path] -factor [percentage of training data e.g. 0.9]
   
    *- path* is tthe same as for processInputFile
    *- factor* a float value indicating the percentage of data for training e.g. 0.9
    As  result of this process two files are generated: *train.txt* with the list of images for training  and *test.txt* with the list of images for testing. These will be required for creating tf_records as indicated previously.
    
# A complete example using MNIST as data
Suppose we are involved in an new interesting project where training a CNN is required. To make this example easier, we will suppose that we are involved in a handwritten recognition problem. To this end, we will use the MNIST dataset, where images can be downloaded from [this link](https://www.dropbox.com/sh/vaumtrc52qkrnnc/AAC827zno64e2G3S0JNyVb2xa?dl=0)

We name MNIST_DIR the path to the mnist data. This path should contain a file indicating the images to process. The file should be named as *list.txt* This has to follow a two-column style, where the first one is the image path and the second  one is the class (it could be a string).

If you have already the test.txt and train.txt files, you can directly got ot step 2., otherwise run steps 1.

## Step 1:  Prepare the data
> python3.6 tools/processInputFile.py -path MNIST_DIR
> python3.6 tools/divideFile.py -path MNIST_DIR -factor 0.8
## Step 2: Create tf_records
> python3.6 tools/create_data.py -type 2 -imheight 40 -imwidth 40 -config [path to config file] -name MNIST

An example of a configuration file is as follows:
```
 [MNIST]
 ARCH = MNIST
 NUM_ITERATIONS = 10000
 NUM_CLASSES = 10
 DATASET_SIZE =  60000
 TEST_SIZE = 10000
 BATCH_SIZE = 92
 SNAPSHOT_TIME = 500
 TEST_TIME = 60
 LEARNING_RATE = 0.0001
 SNAPSHOT_DIR = [path where models will be saved]
 DATA_DIR = [path to data]
 CHANNELS = 1
 ```
 After creating tf_records, please check that MNIST_DIR contains test.tfrecords and train.tfrecords.
## Step 3. Train
> python3.6 tools/train_test_model.py -mode train -device gpu -name MNIST -config [path to config file]
## Step 4 [optional]. Test
> python3.6 tools/train_test_model.py -mode test -device gpu -name MNIST -config [path to config file]
## Step 5 [optional]: Predict
> python3.6 tools/train_test_model.py -mode test -device gpu -name MNIST -config [path to config file] -image [path to image to process]
## Step 6 [optional]: Save the model for faster prediction 
> python3.6 tools/train_test_model.py -mode save -device gpu -name MNIST -config [path to config file] -image [path to image to process]

For faster prediction you will need to run *tools/predict.py*
> python3.6 tools/predict.py -config [path to config file] -name MNIST -list [list of images to process]


# Other Datasets
   - [A small version of mnist](https://www.dropbox.com/sh/se9n4tj3lh35rfm/AACTmKn7F5yVV-SaJbIWgHgna?dl=0), for fast training.
   - [QuicDraw-Animals](https://www.dropbox.com/sh/hsqjv0kd13xda3g/AABYkVk0ruG85s4aL4C1nDKaa?dl=0), including 12 classes of animales from the [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset
   
  Note: For training with a new dataset, you should to appropriately modify  some parameters in the configuration file.
  
**Contact**

*Jose M. Saavedra*

jose\<dot>saavedra\<at>orand\<dot>cl

Orand S.A.
