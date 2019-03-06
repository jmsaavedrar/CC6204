'''
Created on Mar 5, 2019

@author: jsaavedr
'''
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.data as data

if __name__ == '__main__' :
    #datadir = "/home/vision/smb-datasets/SBIR/Quickdraw-10c"
    datadir = "/home/vision/smb-datasets/MNIST-small"
    data.createTFRecord(datadir, 2, (64,64))
    print("tfrecords were created OK")
    
    