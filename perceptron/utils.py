'''
Created on Mar 4, 2019

@author: jsaavedr
'''

def getLineFromWeights(weights,data_min, data_max) :
    """
    A simple function that returns a line defined by two points 
    
    weights: [d + 1 ] weights, the first is the free constant
    data_min: min value of each dimension 
    data_max: max  value of each dimension
    """
    C = weights[0]
    A = weights[1]
    B = weights[2]
     
    p1_x = data_min[0]
    p1_y =  - ( ( A / B ) * p1_x +  ( C / B ) )
    
    p2_x = data_max[0]
    p2_y =  - ( ( A / B ) * p2_x +  ( C / B ) )
        
    
    return [p1_x, p2_x], [p1_y, p2_y]