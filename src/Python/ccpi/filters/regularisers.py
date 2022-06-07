"""
script which assigns a proper device core function based on a flag ('cpu' or 'gpu')
device can also accept an index of a GPU device for multi GPU computing
"""

from ccpi.filters.cpu_regularisers import TV_ROF_CPU, TV_FGP_CPU, TV_PD_CPU, TV_SB_CPU, dTV_FGP_CPU, TNV_CPU, NDF_CPU, Diff4th_CPU, TGV_CPU, LLT_ROF_CPU, PATCHSEL_CPU, NLTV_CPU
try:
    from ccpi.filters.gpu_regularisers import TV_ROF_GPU, TV_FGP_GPU, TV_PD_GPU, TV_SB_GPU, dTV_FGP_GPU, NDF_GPU, Diff4th_GPU, TGV_GPU, LLT_ROF_GPU, PATCHSEL_GPU
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

def _set_gpu_device_index(device):
    GPUdevice_index = -1 # CPU executable
    if gpu_enabled:
        if device == 'gpu':
            GPUdevice_index = 0 # set to 0 the GPU index by default
        else:
            try:
                GPUdevice_index = int(device) # get the GPU index if the integer is given
            except ValueError:
                GPUdevice_index = 0 # roll back to default version
    else:
        if device == 'gpu':
            raise ValueError ('GPU is not available')
    return GPUdevice_index

def ROF_TV(inputData, regularisation_parameter, iterations,
                     time_marching_parameter,tolerance_param, device='cpu'):
    if device == 'cpu':
        return TV_ROF_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param)
    elif device == 'gpu' and gpu_enabled:
        return TV_ROF_GPU(inputData,
                     regularisation_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param)
    else:
        if not gpu_enabled and device == 'gpu':
            raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))

def FGP_TV(inputData, regularisation_parameter,iterations,
                     tolerance_param, methodTV, nonneg, device='cpu'):
    GPUdevice_index = _set_gpu_device_index(device)
    if GPUdevice_index == -1:
        return TV_FGP_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV,
                     nonneg)
    else:
        return TV_FGP_GPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV,
                     nonneg,
                     GPUdevice_index)

def PD_TV(inputData, regularisation_parameter, iterations,
                     tolerance_param, methodTV, nonneg, lipschitz_const, device='cpu'):
    GPUdevice_index = _set_gpu_device_index(device)
    if GPUdevice_index == -1:
        return TV_PD_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV,
                     nonneg,
                     lipschitz_const)
    else:
        return TV_PD_GPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV,
                     nonneg,
                     lipschitz_const,
                     GPUdevice_index)

def SB_TV(inputData, regularisation_parameter, iterations,
                     tolerance_param, methodTV, device='cpu'):
    if device == 'cpu':
        return TV_SB_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV)
    elif device == 'gpu' and gpu_enabled:
        return TV_SB_GPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV)
    else:
        if not gpu_enabled and device == 'gpu':
            raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))
def LLT_ROF(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations,
                     time_marching_parameter, tolerance_param, device='cpu'):
    if device == 'cpu':
        return LLT_ROF_CPU(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, tolerance_param)
    elif device == 'gpu' and gpu_enabled:
        return LLT_ROF_GPU(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, tolerance_param)
    else:
        if not gpu_enabled and device == 'gpu':
            raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))
def TGV(inputData, regularisation_parameter, alpha1, alpha0, iterations,
                     LipshitzConst, tolerance_param, device='cpu'):
    if device == 'cpu':
        return TGV_CPU(inputData,
					regularisation_parameter,
					alpha1,
					alpha0,
					iterations,
                    LipshitzConst,
                    tolerance_param)
    elif device == 'gpu' and gpu_enabled:
        return TGV_GPU(inputData,
					regularisation_parameter,
					alpha1,
					alpha0,
					iterations,
                    LipshitzConst,
                    tolerance_param)
    else:
        if not gpu_enabled and device == 'gpu':
            raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))
def NDF(inputData, regularisation_parameter, edge_parameter, iterations,
                     time_marching_parameter, penalty_type, tolerance_param, device='cpu'):
    if device == 'cpu':
        return NDF_CPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     penalty_type,
                     tolerance_param)
    elif device == 'gpu' and gpu_enabled:
        return NDF_GPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     penalty_type,
                     tolerance_param)
    else:
        if not gpu_enabled and device == 'gpu':
    	    raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))
def Diff4th(inputData, regularisation_parameter, edge_parameter, iterations,
                     time_marching_parameter, tolerance_param, device='cpu'):
    if device == 'cpu':
        return Diff4th_CPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param)
    elif device == 'gpu' and gpu_enabled:
        return Diff4th_GPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param)
    else:
        if not gpu_enabled and device == 'gpu':
            raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))
def FGP_dTV(inputData, refdata, regularisation_parameter, iterations,
                     tolerance_param, eta_const, methodTV, nonneg, device='cpu'):
    if device == 'cpu':
        return dTV_FGP_CPU(inputData,
                     refdata,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     eta_const,
                     methodTV,
                     nonneg)
    elif device == 'gpu' and gpu_enabled:
        return dTV_FGP_GPU(inputData,
                     refdata,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     eta_const,
                     methodTV,
                     nonneg)
    else:
        if not gpu_enabled and device == 'gpu':
            raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))
def TNV(inputData, regularisation_parameter, iterations, tolerance_param):
        return TNV_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param)
def PatchSelect(inputData, searchwindow, patchwindow, neighbours, edge_parameter, device='cpu'):
    if device == 'cpu':
        return PATCHSEL_CPU(inputData,
                     searchwindow,
                     patchwindow,
                     neighbours,
                     edge_parameter)
    elif device == 'gpu' and gpu_enabled:
        return PATCHSEL_GPU(inputData,
                     searchwindow,
                     patchwindow,
                     neighbours,
                     edge_parameter)
    else:
        if not gpu_enabled and device == 'gpu':
            raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))

def NLTV(inputData, H_i, H_j, H_k, Weights, regularisation_parameter, iterations):
    return NLTV_CPU(inputData,
                     H_i,
                     H_j,
                     H_k,
                     Weights,
                     regularisation_parameter,
                     iterations)
