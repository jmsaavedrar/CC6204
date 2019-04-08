#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:48:35 2018

@author: jsaavedr

Class PMapping, that allow us to map a class code  to a class name  in a CNN environment
"""

import os

class PMapping :
    
    def __init__(self, mapping_file):
        """
        constructor
        """
        self.mapping_dict = {}
        self.loadMapping(mapping_file)
    
    
    def loadMapping(self, mapping_file):
        """
        load mapping file for prediction
        """
        assert os.path.exists(mapping_file), "mapping file {} doesn't exist".format(mapping_file)
        self.mapping_dict = {}
        with open(mapping_file) as file:
            lines = [line.rstrip() for line in file]             
            for line in lines :
                sline = line.split('\t')
                self.mapping_dict[int(sline[1])] = sline[0]    
        
    def getClassName(self, class_code) :
        """
        return the class name of a prdicted class, given the class code
        """
        return self.mapping_dict[class_code]
    
    def getClassCode(self, class_name) :        
        """
        return the class code of a given class name for a predicted class
        """
        return list(self.mapping_dict.keys())[list(self.mapping_dict.values()).index(class_name)]