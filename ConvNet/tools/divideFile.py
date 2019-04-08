#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:19:18 2018

@author: divide the input file into train and test files
"""

import argparse
import os
import random
import math

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Divide the input file into train and test")
    parser.add_argument("-path", type = str, help = "<string> input file" , required = True)
    parser.add_argument("-factor", type = float, help = "<float> factor for training > 0.5 & < 1" , required = True)
    pargs = parser.parse_args()
    filename = os.path.join(pargs.path, "list.jfile")
    assert os.path.exists(filename)
    with  open(filename) as file:        
        lines = [line.rstrip() for line in file]     
        random.shuffle(lines)
    pr_train = pargs.factor
    n_total = len(lines)
    n_train = math.floor(n_total * pr_train)
    n_test = n_total - n_train
    lines_train = lines[0:n_train]
    lines_test = lines[n_train::]
    print("{} {} {}".format(len(lines_train), len(lines_test), n_total))
    #saving training dataset
    train_file = os.path.join(os.path.dirname(filename), "train.txt")
    test_file = os.path.join(os.path.dirname(filename), "test.txt")
    f_train = open(train_file, "w+")
    for line in lines_train:
        f_train.write("{}\n".format(line))
    f_train.close()    
    #saving testing dataset
    f_test = open(test_file, "w+")
    for line in lines_test:
        f_test.write("{}\n".format(line))
    f_test.close()
    print("OK")
    
    
    
