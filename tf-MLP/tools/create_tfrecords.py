'''
Created on Mar 5, 2019

@author: jsaavedr
'''
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.data as data
import argparse

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "create tfrecords" )
    parser.add_argument("-path", type = str, help = "path where data is found", required = True)
    input_args = parser.parse_args()
    datadir = input_args.path
    assert os.path.exists(datadir), "{} doesn't exist".format(datadir)
    input_size = 128 # this depends on how the feature vector is computed
    data.createTFRecord(datadir, 2, input_size)
    print("tfrecords were created OK")
    
    