#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:34:32 2018
# quality metrics
@author: algol
"""
import numpy as np

def nrmse(im1, im2):
    a, b = im1.shape
    rmse = np.sqrt(np.sum((im2 - im1) ** 2) / float(a * b))
    max_val = max(np.max(im1), np.max(im2))
    min_val = min(np.min(im1), np.min(im2))
    return 1 - (rmse / (max_val - min_val))
    
def rmse(im1, im2):
    a, b = im1.shape
    rmse = np.sqrt(np.sum((im1 - im2) ** 2) / float(a * b))    
    return rmse