'''
Created on Mar 4, 2019

@author: jsaavedr
This is an example of plotting iris data
'''

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import perceptron
import utils

if __name__ == "__main__" :    
    iris = load_iris()
    data = iris.data[:,[0,3]]
    data = data[0:100]
    target = iris.target[0:100]    
    number_of_epochs = 100000
    learning_rate = 0.01
    w = perceptron.train(data, target, number_of_epochs, learning_rate)
    line_xs, line_ys = utils.getLineFromWeights(w, np.min(data, axis = 0), np.max(data, axis = 0))
    #class 0
    data_0 = data[:50,:]
    #class 1      
    data_1 = data[50:,:]    
    fig, x = plt.subplots(1)
    x.plot(data_0[:,0], data_0[:,1], 'r+' )    
    x.plot(data_1[:,0], data_1[:,1], 'bo' )
    #drawin hyperplane
    l = mlines.Line2D(line_xs, line_ys)
    x.add_line(l)
    x.plot()
    plt.show()
     
    

