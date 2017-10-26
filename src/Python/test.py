# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:08:09 2017

@author: ofn77899
"""

import prova
import numpy as np

a = np.asarray([i for i in range(1*2*3)])
a = a.reshape([1,2,3])
print (a)
b = prova.mexFunction(a)
#print (b)
print (b[4].shape)
print (b[4])
print (b[5])

def print_element(input):
	print ("f: {0}".format(input))
	
prova.doSomething(a, print_element, None)

c = []
def append_to_list(input, shouldPrint=False):
	c.append(input)
	if shouldPrint:
		print ("{0} appended to list {1}".format(input, c))

def element_wise_algebra(input, shouldPrint=True):
	ret = input - 7
	if shouldPrint:
		print ("element_wise {0}".format(ret))
	return ret
		
prova.doSomething(a, append_to_list, None)
#print ("this is c: {0}".format(c))

b = prova.doSomething(a, None, element_wise_algebra)
#print (a)
print (b[5])
