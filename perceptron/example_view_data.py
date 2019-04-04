'''
Created on Mar 4, 2019

@author: jsaavedr
This is an example of plotting iris data
'''

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

def logsig(x):
    """
    sigmoide function
    """
    return 1.0 / ( 1.0 + np.exp( - x ))

def tanh(x):    
    return (np.exp(x )- np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dtanh(x):    
    a = (np.exp(x )- np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return 1- a*a

def dlogsig(x):
    return np.exp( - x ) / (( 1.0 + np.exp( - x ))**2)

def loadIrisData():
    iris = load_iris()
    data = iris.data[:,[0,3]]
    data = data[0:100]
    target = iris.target[0:100]    
    #data for class 0 
    data_0 = data[:50,:]
    #data for class 1      
    data_1 = data[50:,:]    
    fig, x = plt.subplots(1)
    x.set_xlabel("petal length")
    x.set_ylabel("sepal width")
    x.plot(data_0[:,0], data_0[:,1], 'r+' )    
    x.plot(data_1[:,0], data_1[:,1], 'bo' )
    x.plot()
    plt.show()

def plotFunction(fun):
    x = np.arange(-1,1,0.01)
    print(x)
    y = fun(x)
    fig, gx = plt.subplots(1)
    gx.plot(x,y, 'bo')
    plt.show()
      
if __name__ == "__main__" :             
    plotFunction(dlogsig)
