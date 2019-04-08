#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:56:25 2018

@author: jsaavedr

"""
import tensorflow as tf
import sys
if __name__ == '__main__':  
    filename= sys.argv[1]
    for idx, event in enumerate(tf.train.summary_iterator(filename)) :
        print("event #{}".format(idx))
        for value in event.summary.value:
            if value.HasField('simple_value'):
                val = value.simple_value
            print("---{}: {}".format(value.tag, val))