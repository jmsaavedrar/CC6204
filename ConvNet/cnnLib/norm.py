#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:50:16 2018

@author: jsaavedr

Differetn method for normalization
"""
import numpy as np

#square root normalization
def squareRootNorm(data):    
    #square_root
    #add 1 to avoit division by zero
    sign_minus = data < 0
    data_abs = np.abs(data)
    data_sqrt = np.sqrt(data_abs)
    data_sqrt[sign_minus] = data_sqrt[sign_minus] * -1
    
    #
    if len(data.shape) > 1 :        
        dim = data.shape[1]
        norm2 = np.sqrt(np.sum(np.square(data_sqrt), axis =1))            
        norm2_r = np.repeat(norm2, dim, axis = 0)        
        norm2_r = np.reshape(norm2_r, [-1,dim])
    else:        
        dim = data.size
        norm2 = np.sqrt(np.sum(np.square(data_sqrt)))        
        norm2_r = np.repeat(norm2, dim, axis = 0)
        norm2_r = np.reshape(norm2_r, [dim])
          
    normed_data = data_sqrt /  norm2_r
            
    return normed_data


if __name__ == '__main__' :
    #A little test 
    a =np.array([[1,2,3], [2,10,6]])
    nn = np.sqrt(np.sum(a**2))
    print(nn)
    b= squareRootNorm(a)
    print(a)
    print(b)