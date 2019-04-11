"""
Author: jose.saavedra
This is an example of using convolution with gabor filters
"""

import skimage.filters as filters
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import skimage.io as io


if __name__ == '__main__' :
    filename = "images/chair.jpg"
    image = io.imread(filename, as_gray = True) 
    gk1 = filters.gabor_kernel(frequency=0.1,  theta = 0)
    #try wiht gk1 = filters.gabor_kernel(frequency=0.1,  theta = np.pi/2)        
    imageg = ndimage.convolve(image, gk1.real)    
    fig, x = plt.subplots(1,2)
    x[0].imshow(imageg, cmap = 'gray')
    x[1].imshow(gk1.real, cmap = 'gray')
    plt.show()
