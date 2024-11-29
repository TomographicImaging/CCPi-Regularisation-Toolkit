import ctypes
import numpy as np
from .utils import cilreg, cilregcuda

NDF_GPU = None
Diffus4th_GPU = None


def NDF_CPU(
    inputData,
    lambdaPar,
    sigmaPar,
    iterationsNumb,
    tau,
    penaltytype,
    epsil,
    out=None,
    infovector=None,
):
    # float Diffusion_CPU_main(float *Input, float *Output, float *infovector, float lambdaPar,
    # float sigmaPar, int iterationsNumb, float tau, int penaltytype, float epsil, int dimX, int dimY, int dimZ);
    cilreg.Diffusion_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # lambdaPar (float)
        ctypes.c_float,  # sigmaPar (float)
        ctypes.c_int,  # iterationsNumb (int)
        ctypes.c_float,  # tau (float)
        ctypes.c_int,  # penaltytype (int)
        ctypes.c_float,  # epsil (float)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.Diffusion_CPU_main.restype = ctypes.c_float  # return value is float

    in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if out is None:
        out = np.zeros_like(inputData)
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if infovector is None:
        infovector = np.zeros((2,), dtype="float32")
    infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dims = list(inputData.shape)[::-1]
    if inputData.ndim == 2:
        dims.append(1)

    # float Diffusion_CPU_main(float *Input, float *Output, float *infovector, float lambdaPar,
    # float sigmaPar, int iterationsNumb, float tau, int penaltytype, float epsil, int dimX, int dimY, int dimZ);
    cilreg.Diffusion_CPU_main(
        in_p,
        out_p,
        infovector_p,
        lambdaPar,
        sigmaPar,
        iterationsNumb,
        tau,
        penaltytype,
        epsil,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


def Diffus4th_CPU(
    inputData,
    lambdaPar,
    sigmaPar,
    iterationsNumb,
    tau,
    epsil,
    out=None,
    infovector=None,
):
    # float Diffus4th_CPU_main(float *Input, float *Output,  float *infovector, float lambdaPar,
    # float sigmaPar, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ);
    cilreg.Diffus4th_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # lambdaPar (float)
        ctypes.c_float,  # sigmaPar (float)
        ctypes.c_int,  # iterationsNumb (int)
        ctypes.c_float,  # tau (float)
        ctypes.c_float,  # epsil (float)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.Diffus4th_CPU_main.restype = ctypes.c_float  # return value is float

    in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if out is None:
        out = np.zeros_like(inputData)
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if infovector is None:
        infovector = np.zeros((2,), dtype="float32")
    infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dims = list(inputData.shape)[::-1]
    if inputData.ndim == 2:
        dims.append(1)

    # float Diffus4th_CPU_main(float *Input, float *Output,  float *infovector, float lambdaPar,
    # float sigmaPar, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ);
    cilreg.Diffus4th_CPU_main(
        in_p,
        out_p,
        infovector_p,
        lambdaPar,
        sigmaPar,
        iterationsNumb,
        tau,
        epsil,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


if cilregcuda is not None:

    def NDF_GPU(
        inputData,
        lambdaPar,
        sigmaPar,
        iterationsNumb,
        tau,
        penaltytype,
        epsil,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int NonlDiff_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float sigmaPar,
        # int iterationsNumb, float tau, int penaltytype, float epsil, int gpu_device, int N, int M, int Z);
        cilregcuda.NonlDiff_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_float,  # sigmaPar (float)
            ctypes.c_int,  # iterationsNumb (int)
            ctypes.c_float,  # tau (float)
            ctypes.c_int,  # penaltytype (int)
            ctypes.c_float,  # epsil (float)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # Z (int)
        ]
        cilregcuda.NonlDiff_GPU_main.restype = ctypes.c_int  # return value is int

        in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if out is None:
            out = np.zeros_like(inputData)
        out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if infovector is None:
            infovector = np.zeros((2,), dtype="float32")
        infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        dims = list(inputData.shape)[::-1]
        if inputData.ndim == 2:
            dims.append(1)

        # int NonlDiff_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float sigmaPar,
        # int iterationsNumb, float tau, int penaltytype, float epsil, int gpu_device, int N, int M, int Z);
        result = cilregcuda.NonlDiff_GPU_main(
            in_p,
            out_p,
            infovector_p,
            lambdaPar,
            sigmaPar,
            iterationsNumb,
            tau,
            penaltytype,
            epsil,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out

    def Diffus4th_GPU(
        inputData,
        lambdaPar,
        sigmaPar,
        iterationsNumb,
        tau,
        epsil,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int Diffus4th_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float sigmaPar,
        # int iterationsNumb, float tau, float epsil, int gpu_device, int N, int M, int Z);
        cilregcuda.Diffus4th_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_float,  # sigmaPar (float)
            ctypes.c_int,  # iterationsNumb (int)
            ctypes.c_float,  # tau (float)
            ctypes.c_float,  # epsil (float)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # Z (int)
        ]
        cilregcuda.Diffus4th_GPU_main.restype = ctypes.c_int  # return value is int

        in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if out is None:
            out = np.zeros_like(inputData)
        out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if infovector is None:
            infovector = np.zeros((2,), dtype="float32")
        infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        dims = list(inputData.shape)[::-1]
        if inputData.ndim == 2:
            dims.append(1)

        # int Diffus4th_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float sigmaPar,
        # int iterationsNumb, float tau, float epsil, int gpu_device, int N, int M, int Z);
        result = cilregcuda.Diffus4th_GPU_main(
            in_p,
            out_p,
            infovector_p,
            lambdaPar,
            sigmaPar,
            iterationsNumb,
            tau,
            epsil,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out
