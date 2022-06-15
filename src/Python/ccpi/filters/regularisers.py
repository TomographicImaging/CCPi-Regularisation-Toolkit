""" script which assigns a proper device core function based on a flag ('cpu', 'gpu' or GPU integer)
device can also accept an index of a GPU device for multi GPU computing.
"""

from ccpi.filters.cpu_regularisers import TV_ROF_CPU, TV_FGP_CPU, TV_PD_CPU, TV_SB_CPU, dTV_FGP_CPU, TNV_CPU, NDF_CPU, Diff4th_CPU, TGV_CPU, LLT_ROF_CPU, PATCHSEL_CPU, NLTV_CPU
try:
    from ccpi.filters.gpu_regularisers import TV_ROF_GPU, TV_FGP_GPU, TV_PD_GPU, TV_SB_GPU, dTV_FGP_GPU, NDF_GPU, Diff4th_GPU, TGV_GPU, LLT_ROF_GPU, PATCHSEL_GPU
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

def parse_device_argument(device_int_or_string, gpu_enabled):
    """Convert a cpu/gpu string or integer gpu number into a tuple."""
    if isinstance(device_int_or_string, int):
        return "gpu", device_int_or_string
    elif device_int_or_string == 'gpu':
        return "gpu", 0
    elif device_int_or_string == 'cpu':
        return "cpu", -1
    else:
        raise ValueError('Unknown device {0}. Expecting either "cpu" or "gpu" strings OR index of a gpu device'\
                         .format(device_int_or_string))    

def ROF_TV(inputData, regularisation_parameter, iterations,
                     time_marching_parameter,tolerance_param, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return TV_ROF_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param)
    else:
        return TV_ROF_GPU(inputData,
                     regularisation_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param,
                     GPUdevice_index)

def FGP_TV(inputData, regularisation_parameter,iterations,
                     tolerance_param, methodTV, nonneg, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
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
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
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
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return TV_SB_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV)
    else:
        return TV_SB_GPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     methodTV,
                     GPUdevice_index)

def LLT_ROF(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations,
                     time_marching_parameter, tolerance_param, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return LLT_ROF_CPU(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, tolerance_param)
    else:
        return LLT_ROF_GPU(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, tolerance_param, GPUdevice_index)    
def TGV(inputData, regularisation_parameter, alpha1, alpha0, iterations,
                     LipshitzConst, tolerance_param, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return TGV_CPU(inputData,
					regularisation_parameter,
					alpha1,
					alpha0,
					iterations,
                    LipshitzConst,
                    tolerance_param)
    else:
        return TGV_GPU(inputData,
					regularisation_parameter,
					alpha1,
					alpha0,
					iterations,
                    LipshitzConst,
                    tolerance_param,
                    GPUdevice_index)

def NDF(inputData, regularisation_parameter, edge_parameter, iterations,
                     time_marching_parameter, penalty_type, tolerance_param, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return NDF_CPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     penalty_type,
                     tolerance_param)
    else:
        return NDF_GPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     penalty_type,
                     tolerance_param,
                     GPUdevice_index)

def Diff4th(inputData, regularisation_parameter, edge_parameter, iterations,
                     time_marching_parameter, tolerance_param, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return Diff4th_CPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param)
    else:
        return Diff4th_GPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations,
                     time_marching_parameter,
                     tolerance_param,
                     GPUdevice_index)

def FGP_dTV(inputData, refdata, regularisation_parameter, iterations,
                     tolerance_param, eta_const, methodTV, nonneg, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return dTV_FGP_CPU(inputData,
                     refdata,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     eta_const,
                     methodTV,
                     nonneg)
    else:
        return dTV_FGP_GPU(inputData,
                     refdata,
                     regularisation_parameter,
                     iterations,
                     tolerance_param,
                     eta_const,
                     methodTV,
                     nonneg,
                     GPUdevice_index)

def TNV(inputData, regularisation_parameter, iterations, tolerance_param):
        return TNV_CPU(inputData,
                     regularisation_parameter,
                     iterations,
                     tolerance_param)
def PatchSelect(inputData, searchwindow, patchwindow, neighbours, edge_parameter, device='cpu'):
    device, GPUdevice_index = parse_device_argument(device, gpu_enabled)
    if GPUdevice_index == -1:
        return PATCHSEL_CPU(inputData,
                     searchwindow,
                     patchwindow,
                     neighbours,
                     edge_parameter)
    else:
        return PATCHSEL_GPU(inputData,
                     searchwindow,
                     patchwindow,
                     neighbours,
                     edge_parameter,
                     GPUdevice_index)

def NLTV(inputData, H_i, H_j, H_k, Weights, regularisation_parameter, iterations):
    return NLTV_CPU(inputData,
                     H_i,
                     H_j,
                     H_k,
                     Weights,
                     regularisation_parameter,
                     iterations)
