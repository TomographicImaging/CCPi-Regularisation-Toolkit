"""
script which assigns a proper device core function based on a flag ('cpu' or 'gpu')
"""

from ccpi.filters.cpu_regularizers_cython import TV_ROF_CPU, TV_FGP_CPU
from ccpi.filters.gpu_regularizers import TV_ROF_GPU, TV_FGP_GPU

def ROF_TV(inputData, regularization_parameter, iterations,
                     time_marching_parameter,device='cpu'):
    if device == 'cpu':
        return TV_ROF_CPU(inputData,
                     regularization_parameter,
                     iterations, 
                     time_marching_parameter)
    elif device == 'gpu':
        return TV_ROF_GPU(inputData,
                     regularization_parameter,
                     iterations, 
                     time_marching_parameter)
    else:
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))

def FGP_TV(inputData, regularization_parameter,iterations,
                     tolerance_param, methodTV, nonneg, printM, device='cpu'):
    if device == 'cpu':
        return TV_FGP_CPU(inputData,
                     regularization_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     nonneg,
                     printM)
    elif device == 'gpu':
        return TV_FGP_GPU(inputData,
                     regularization_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     nonneg,
                     printM)
    else:
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))