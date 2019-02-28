'''
Created on Feb 28, 2019

@author: jsaavedr
'''

import  sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import basic_lib.perceptron as perceptron

if __name__ == "__main__" :    
    iris = load_iris()
    data = iris.data[:,[1,2]]
    data = data[0:100]
    target = iris.target[0:100]    
    
    w = perceptron.train(data, target, 100000, 0.01)
    out = perceptron.predict(data, w)
    print(out)
    print(target)
#     data_0 = data[:50,:]      
#     data_1 = data[50:,:]    
#     fig, x = plt.subplots(1)
#     x.plot(data_0[:,0], data_0[:,1], 'r+' )    
#     x.plot(data_1[:,0], data_1[:,1], 'bo' )
#     plt.show()
     
    

