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
    datadir = "/home/vision/smb-datasets/SBIR/QuickDraw-10c"
    data.createTFRecord(datadir, 2, 128)
    print("tfrecords were created OK")
    
    