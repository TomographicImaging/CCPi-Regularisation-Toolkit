# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:08:09 2017

@author: ofn77899
"""

import fista
import numpy as np

a = np.asarray([i for i in range(3*4*5)])
a = a.reshape([3,4,5])
print (a)
b = fista.mexFunction(a)
#print (b)
print (b[4].shape)
print (b[4])
print (b[5])