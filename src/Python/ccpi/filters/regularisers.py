import numbers
from functools import wraps

# CPU regularisers
from .TV import TV_ROF_CPU, TV_ROF_GPU
from .TV import TV_FGP_CPU, TV_FGP_GPU
from .TV import PDTV_CPU, PDTV_GPU
from .TV import SB_TV_CPU, SB_TV_GPU
from .TV import LLT_ROF_CPU, LLT_ROF_GPU
from .TV import TGV_CPU, TGV_GPU
from .TV import dTV_FGP_CPU, dTV_FGP_GPU
from .TV import PatchSelect_CPU, PatchSelect_GPU


from .diffusion import NDF_GPU, NDF_CPU
from .diffusion import Diffus4th_GPU, Diffus4th_CPU
from .utils import cilregcuda


def create_wrapper(CPU_func, GPU_func):
    @wraps(CPU_func)
    def wrapper(*args, device="cpu", **kwargs):
        if device == "cpu":
            return CPU_func(*args, **kwargs)
        elif (
            device == "gpu"
            or isinstance(device, numbers.Integral)
            and cilregcuda is not None
        ):
            return GPU_func(
                *args, gpu_device=0 if device == "gpu" else device, **kwargs
            )
        raise KeyError(f"{GPU_func.__name__}: device {device} not available")

    return wrapper


# TODO: The functions below will not have a docstring.
# This could be added as described here
# https://stackoverflow.com/questions/53564301/insert-docstring-attributes-in-a-python-file
from .TV import NLTV
from .TV import TV_ENERGY
from .TV import TNV

ROF_TV = create_wrapper(TV_ROF_CPU, TV_ROF_GPU)
FGP_TV = create_wrapper(TV_FGP_CPU, TV_FGP_GPU)
PD_TV = create_wrapper(PDTV_CPU, PDTV_GPU)
SB_TV = create_wrapper(SB_TV_CPU, SB_TV_GPU)
PD_TV = create_wrapper(PDTV_CPU, PDTV_GPU)
LLT_ROF = create_wrapper(LLT_ROF_CPU, LLT_ROF_GPU)
TGV = create_wrapper(TGV_CPU, TGV_GPU)
FGP_dTV = create_wrapper(dTV_FGP_CPU, dTV_FGP_GPU)

NDF = create_wrapper(NDF_CPU, NDF_GPU)
Diff4th = create_wrapper(Diffus4th_CPU, Diffus4th_GPU)

PatchSelect = create_wrapper(PatchSelect_CPU, PatchSelect_GPU)
