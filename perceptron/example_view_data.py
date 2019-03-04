'''
Created on Mar 4, 2019

@author: jsaavedr
This is an example of plotting iris data
'''

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

if __name__ == "__main__" :    
    iris = load_iris()
    data = iris.data[:,[0,3]]
    data = data[0:100]
    target = iris.target[0:100]    
    #data for class 0 
    data_0 = data[:50,:]
    #data for class 1      
    data_1 = data[50:,:]    
    fig, x = plt.subplots(1)
    x.plot(data_0[:,0], data_0[:,1], 'r+' )    
    x.plot(data_1[:,0], data_1[:,1], 'bo' )
    x.plot()
    plt.show()
     
    

