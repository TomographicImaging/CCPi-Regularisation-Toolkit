import ctypes
import numpy as np
from .utils import cilreg, cilregcuda


# %%


def TV_ROF_CPU(
    inputData,
    regularisation_parameter,
    iterationsNumb,
    marching_step_parameter,
    tolerance_param,
    out=None,
    infovector=None,
):
    # float TV_ROF_CPU_main(float *Input, float *Output, float *infovector,
    #       float *lambdaPar, int lambda_is_arr, int iterationsNumb, float tau,
    #       float epsil, int dimX, int dimY, int dimZ);
    cilreg.TV_ROF_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infoVector array
        ctypes.POINTER(ctypes.c_float),  # type of type of lambdaPar (float)
        ctypes.c_int,  # lambda_is_arr (int)
        ctypes.c_int,  # type of type of iterationsNumb (int)
        ctypes.c_float,  # type of type of tau (float)
        ctypes.c_float,  # type of type of epsil (float)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.TV_ROF_CPU_main.restype = ctypes.c_float  # return value is float

    in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if out is None:
        out = inputData * 0
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if infovector is None:
        infovector = np.zeros((2,), dtype="float32")
    infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dims = list(inputData.shape)[::-1]
    if inputData.ndim == 2:
        dims.append(1)

    # float TV_ROF_CPU_main(float *Input, float *Output, float *infovector,
    #       float *lambdaPar, int lambda_is_arr, int iterationsNumb, float tau,
    #       float epsil, int dimX, int dimY, int dimZ);
    cilreg.TV_ROF_CPU_main(
        in_p,
        out_p,
        infovector_p,
        ctypes.byref(ctypes.c_float(regularisation_parameter)),
        0,
        iterationsNumb,
        marching_step_parameter,
        tolerance_param,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


# %% From copilot chat
def TV_FGP_CPU(
    inputData,
    lambdaPar,
    iterationsNumb,
    epsil,
    methodTV,
    nonneg,
    out=None,
    infovector=None,
):
    # float TV_FGP_CPU_main(float *Input, float *Output, float *infovector, float lambdaPar,
    # int iterationsNumb, float epsil, int methodTV, int nonneg, int dimX, int dimY, int dimZ);
    cilreg.TV_FGP_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # lambdaPar (float)
        ctypes.c_int,  # iterationsNumb (int)
        ctypes.c_float,  # epsil (float)
        ctypes.c_int,  # methodTV (int)
        ctypes.c_int,  # nonneg (int)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.TV_FGP_CPU_main.restype = ctypes.c_float  # return value is float

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

    # float TV_FGP_CPU_main(float *Input, float *Output, float *infovector, float lambdaPar,
    # int iterationsNumb, float epsil, int methodTV, int nonneg, int dimX, int dimY, int dimZ);
    cilreg.TV_FGP_CPU_main(
        in_p,
        out_p,
        infovector_p,
        lambdaPar,
        iterationsNumb,
        epsil,
        methodTV,
        nonneg,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


def PDTV_CPU(
    inputData,
    lambdaPar,
    iterationsNumb,
    epsil,
    lipschitz_const,
    methodTV,
    nonneg,
    out=None,
    infovector=None,
):
    # float PDTV_CPU_main(float *Input, float *U, float *infovector,
    # float lambdaPar, int iterationsNumb, float epsil,
    # float lipschitz_const, int methodTV, int nonneg,
    # int dimX, int dimY, int dimZ);

    cilreg.PDTV_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the U array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # lambdaPar (float)
        ctypes.c_int,  # iterationsNumb (int)
        ctypes.c_float,  # epsil (float)
        ctypes.c_float,  # lipschitz_const (float)
        ctypes.c_int,  # methodTV (int)
        ctypes.c_int,  # nonneg (int)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.PDTV_CPU_main.restype = ctypes.c_float  # return value is float

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

    # float PDTV_CPU_main(float *Input, float *U, float *infovector, float lambdaPar,
    # int iterationsNumb, float epsil, float lipschitz_const, int methodTV, int nonneg,
    # int dimX, int dimY, int dimZ);
    cilreg.PDTV_CPU_main(
        in_p,
        out_p,
        infovector_p,
        lambdaPar,
        iterationsNumb,
        epsil,
        lipschitz_const,
        methodTV,
        nonneg,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


def SB_TV_CPU(inputData, mu, iter, epsil, methodTV, out=None, infovector=None):
    # float SB_TV_CPU_main(float *Input, float *Output, float *infovector, float mu,
    # int iter, float epsil, int methodTV, int dimX, int dimY, int dimZ);
    cilreg.SB_TV_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # mu (float)
        ctypes.c_int,  # iter (int)
        ctypes.c_float,  # epsil (float)
        ctypes.c_int,  # methodTV (int)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.SB_TV_CPU_main.restype = ctypes.c_float  # return value is float

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

    # float SB_TV_CPU_main(float *Input, float *Output, float *infovector, float mu,
    # int iter, float epsil, int methodTV, int dimX, int dimY, int dimZ);
    cilreg.SB_TV_CPU_main(
        in_p, out_p, infovector_p, mu, iter, epsil, methodTV, dims[0], dims[1], dims[2]
    )

    return out


def LLT_ROF_CPU(
    inputData,
    lambdaROF,
    lambdaLLT,
    iterationsNumb,
    tau,
    epsil,
    out=None,
    infovector=None,
):
    # float LLT_ROF_CPU_main(float *Input, float *Output, float *infovector, float lambdaROF,
    # float lambdaLLT, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ);
    cilreg.LLT_ROF_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # lambdaROF (float)
        ctypes.c_float,  # lambdaLLT (float)
        ctypes.c_int,  # iterationsNumb (int)
        ctypes.c_float,  # tau (float)
        ctypes.c_float,  # epsil (float)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.LLT_ROF_CPU_main.restype = ctypes.c_float  # return value is float

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

    # float LLT_ROF_CPU_main(float *Input, float *Output, float *infovector, float lambdaROF,
    # float lambdaLLT, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ);
    cilreg.LLT_ROF_CPU_main(
        in_p,
        out_p,
        infovector_p,
        lambdaROF,
        lambdaLLT,
        iterationsNumb,
        tau,
        epsil,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


# TGV
def TGV_CPU(
    inputData,
    lambdaPar,
    alpha1,
    alpha0,
    iterationsNumb,
    L2,
    epsil,
    out=None,
    infovector=None,
):
    # float TGV_main(float *Input, float *Output, float *infovector, float lambdaPar,
    # float alpha1, float alpha0, int iterationsNumb, float L2, float epsil, int dimX, int dimY, int dimZ);
    cilreg.TGV_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # lambdaPar (float)
        ctypes.c_float,  # alpha1 (float)
        ctypes.c_float,  # alpha0 (float)
        ctypes.c_int,  # iterationsNumb (int)
        ctypes.c_float,  # L2 (float)
        ctypes.c_float,  # epsil (float)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.TGV_main.restype = ctypes.c_float  # return value is float

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

    # float TGV_main(float *Input, float *Output, float *infovector, float lambdaPar,
    # float alpha1, float alpha0, int iterationsNumb, float L2, float epsil, int dimX, int dimY, int dimZ);
    cilreg.TGV_main(
        in_p,
        out_p,
        infovector_p,
        lambdaPar,
        alpha1,
        alpha0,
        iterationsNumb,
        L2,
        epsil,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


# dTV
def dTV_FGP_CPU(
    inputData,
    inputRef,
    lambdaPar,
    iterationsNumb,
    epsil,
    eta,
    methodTV,
    nonneg,
    out=None,
    infovector=None,
):
    # dTV_FGP_CPU_main(float *Input, float *InputRef, float *Output, float *infovector, float lambdaPar,
    # int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int dimX, int dimY, int dimZ);
    cilreg.dTV_FGP_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the InputRef array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
        ctypes.c_float,  # lambdaPar (float)
        ctypes.c_int,  # iterationsNumb (int)
        ctypes.c_float,  # epsil (float)
        ctypes.c_float,  # eta (float)
        ctypes.c_int,  # methodTV (int)
        ctypes.c_int,  # nonneg (int)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.dTV_FGP_CPU_main.restype = ctypes.c_float  # return value is float

    in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    inref_p = inputRef.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if out is None:
        out = np.zeros_like(inputData)
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if infovector is None:
        infovector = np.zeros((2,), dtype="float32")
    infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dims = list(inputData.shape)[::-1]
    if inputData.ndim == 2:
        dims.append(1)

    # dTV_FGP_CPU_main(float *Input, float *InputRef, float *Output, float *infovector, float lambdaPar,
    # int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int dimX, int dimY, int dimZ);
    cilreg.dTV_FGP_CPU_main(
        in_p,
        inref_p,
        out_p,
        infovector_p,
        lambdaPar,
        iterationsNumb,
        epsil,
        eta,
        methodTV,
        nonneg,
        dims[0],
        dims[1],
        dims[2],
    )

    return out


# TNV
def TNV(inputData, lambdaPar, maxIter, tol, out=None):
    # float TNV_CPU_main(float *Input, float *u, float lambdaPar, int maxIter, float tol, int dimX, int dimY, int dimZ);
    cilreg.TNV_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_float),  # pointer to the u array
        ctypes.c_float,  # lambdaPar (float)
        ctypes.c_int,  # maxIter (int)
        ctypes.c_float,  # tol (float)
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
    ]
    cilreg.TNV_CPU_main.restype = ctypes.c_float  # return value is float

    in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if out is None:
        out = np.zeros_like(inputData)
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dims = list(inputData.shape)[::-1]
    if inputData.ndim != 3:
        raise ValueError("Input data must be 2D + 1 channel")

    # float TNV_CPU_main(float *Input, float *u, float lambdaPar, int maxIter, float tol, int dimX, int dimY, int dimZ);
    cilreg.TNV_CPU_main(in_p, out_p, lambdaPar, maxIter, tol, dims[0], dims[1], dims[2])

    return out


# Non-local TV
# usage example https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/blob/71f8d304d804b54d378f0ed05539f01aaaf13758/demos/demo_gpu_regularisers.py#L438-L506
def PatchSelect_CPU(
    inputData,
    SearchWindow,
    SimilarWin,
    NumNeighb,
    h,
    H_i=None,
    H_j=None,
    H_k=None,
    Weights=None,
):
    # float PatchSelect_CPU_main(float *Input, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k,
    # float *Weights, int dimX, int dimY, int dimZ, int SearchWindow, int SimilarWin, int NumNeighb, float h);
    cilreg.PatchSelect_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
        ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_i array
        ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_j array
        ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_k array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Weights array
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
        ctypes.c_int,  # SearchWindow (int)
        ctypes.c_int,  # SimilarWin (int)
        ctypes.c_int,  # NumNeighb (int)
        ctypes.c_float,  # h (float)
    ]
    cilreg.PatchSelect_CPU_main.restype = ctypes.c_float  # return value is float
    if inputData.ndim != 2:
        # See https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/issues/184
        raise ValueError("PatchSelect_CPU: Only 2D images are supported")
    dims = [NumNeighb, inputData.shape[0], inputData.shape[1]]

    if Weights is None:
        Weights = np.zeros(dims, dtype="float32")
    if H_i is None:
        H_i = np.zeros(dims, dtype="uint16")
    if H_j is None:
        H_j = np.zeros(dims, dtype="uint16")
    if H_k is None:
        H_k = np.zeros(dims, dtype="uint16")

    in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hi_p = H_i.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    hj_p = H_j.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    hk_p = H_k.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    weights_p = Weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # float PatchSelect_CPU_main(float *Input, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k,
    # float *Weights, int dimX, int dimY, int dimZ, int SearchWindow, int SimilarWin, int NumNeighb, float h);
    cilreg.PatchSelect_CPU_main(
        in_p,
        hi_p,
        hj_p,
        hk_p,
        weights_p,
        dims[2],
        dims[1],
        0,
        SearchWindow,
        SimilarWin,
        NumNeighb,
        h,
    )

    return H_i, H_j, Weights


def NLTV(
    inputData,
    H_i,
    H_j,
    H_k,
    Weights,
    NumNeighb,
    lambdaReg,
    IterNumb,
    switchM=1,
    Output=None,
):
    # switchM=1 is the default value
    # H_k is not used as only 2D and it is passed as H_i to the C function
    # see below link for the original code
    # https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/blob/71f8d304d804b54d378f0ed05539f01aaaf13758/src/Python/src/cpu_regularisers.pyx#L689

    # float Nonlocal_TV_CPU_main(float *A_orig, float *Output, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k,
    # float *Weights, int dimX, int dimY, int dimZ, int NumNeighb, float lambdaReg, int IterNumb, int switchM);
    cilreg.Nonlocal_TV_CPU_main.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pointer to the A_orig array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
        ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_i array
        ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_j array
        ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_k array
        ctypes.POINTER(ctypes.c_float),  # pointer to the Weights array
        ctypes.c_int,  # dimX (int)
        ctypes.c_int,  # dimY (int)
        ctypes.c_int,  # dimZ (int)
        ctypes.c_int,  # NumNeighb (int)
        ctypes.c_float,  # lambdaReg (float)
        ctypes.c_int,  # IterNumb (int)
        ctypes.c_int,  # switchM (int)
    ]
    cilreg.Nonlocal_TV_CPU_main.restype = ctypes.c_float  # return value is float

    dims = list(inputData.shape)
    if inputData.ndim != 2:
        raise ValueError(
            f"NLTV can only process 2D data. Got {inputData.ndim} dimensions"
        )

    aorig_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hi_p = H_i.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    hj_p = H_j.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    # hk_p = H_k.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    hk_p = hi_p
    weights_p = Weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if Output is None:
        Output = np.zeros_like(inputData)
    out_p = Output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # float Nonlocal_TV_CPU_main(float *A_orig, float *Output, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k,
    # float *Weights, int dimX, int dimY, int dimZ, int NumNeighb, float lambdaReg, int IterNumb, int switchM);
    result = cilreg.Nonlocal_TV_CPU_main(
        aorig_p,
        out_p,
        hi_p,
        hj_p,
        hk_p,
        weights_p,
        dims[1],
        dims[0],
        0,
        NumNeighb,
        lambdaReg,
        IterNumb,
        switchM,
    )

    return Output


def TV_ENERGY(U, U0, lambdaPar, type, E_val=None):
    u_p = U.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    u0_p = U0.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    E_val = np.zeros([1], dtype="float32")
    e_val_p = E_val.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dims = list(U.shape)

    if U.ndim == 2:
        # float TV_energy2D(float *U, float *U0, float *E_val, float lambdaPar, int type, int dimX, int dimY);
        cilreg.TV_energy2D.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the U array
            ctypes.POINTER(ctypes.c_float),  # pointer to the U0 array
            ctypes.POINTER(ctypes.c_float),  # pointer to the E_val array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_int,  # type (int)
            ctypes.c_int,  # dimX (int)
            ctypes.c_int,  # dimY (int)
        ]
        cilreg.TV_energy2D.restype = ctypes.c_float  # return value is float
        result = cilreg.TV_energy2D(
            u_p, u0_p, e_val_p, lambdaPar, type, dims[1], dims[0]
        )
    elif U.ndim == 3:
        # float TV_energy3D(float *U, float *U0, float *E_val, float lambdaPar, int type, int dimX, int dimY, int dimZ);
        cilreg.TV_energy3D.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the U array
            ctypes.POINTER(ctypes.c_float),  # pointer to the U0 array
            ctypes.POINTER(ctypes.c_float),  # pointer to the E_val array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_int,  # type (int)
            ctypes.c_int,  # dimX (int)
            ctypes.c_int,  # dimY (int)
            ctypes.c_int,  # dimZ (int)
        ]
        cilreg.TV_energy3D.restype = ctypes.c_float  # return value is float

        # float TV_energy3D(float *U, float *U0, float *E_val, float lambdaPar, int type, int dimX, int dimY, int dimZ);
        result = cilreg.TV_energy3D(
            u_p, u0_p, e_val_p, lambdaPar, type, dims[2], dims[1], dims[0]
        )
    else:
        raise ValueError(f"TV_ENERGY: Only 2D and 3D data are supported. Got {U.ndim}")
    return E_val


# Define all the GPU functions as None
TV_FGP_GPU = None
TV_ROF_GPU = None
PDTV_GPU = None
SB_TV_GPU = None
LLT_ROF_GPU = None

TGV_GPU = None
dTV_FGP_GPU = None

PatchSelect_GPU = None

if cilregcuda is not None:

    def TV_ROF_GPU(
        inputData,
        regularisation_parameter,
        iterationsNumb,
        marching_step_parameter,
        tolerance_param,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int TV_ROF_GPU_main(float* Input, float* Output, float *infovector,
        #                     float lambdaPar, int iter, float tau, float epsil, int gpu_device,
        #                     int N, int M, int Z);
        cilregcuda.TV_ROF_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infoVector array
            ctypes.c_float,  # type of type of lambdaPar (float)
            ctypes.c_int,  # type of type of iterationsNumb (int)
            ctypes.c_float,  # type of type of tau (float)
            ctypes.c_float,  # type of type of epsil (float)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # Z (int)
        ]
        cilreg.TV_ROF_CPU_main.restype = ctypes.c_float  # return value is float

        in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if out is None:
            out = inputData * 0
        out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if infovector is None:
            infovector = np.zeros((2,), dtype="float32")
        infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        dims = list(inputData.shape)[::-1]
        if inputData.ndim == 2:
            dims.append(1)

        # float TV_ROF_CPU_main(float *Input, float *Output, float *infovector,
        #       float *lambdaPar, int lambda_is_arr, int iterationsNumb, float tau,
        #       float epsil, int dimX, int dimY, int dimZ);
        if gpu_device == "gpu":
            gpu_device = 0
        cilregcuda.TV_ROF_GPU_main(
            in_p,
            out_p,
            infovector_p,
            regularisation_parameter,
            iterationsNumb,
            marching_step_parameter,
            tolerance_param,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out

    def TV_FGP_GPU(
        inputData,
        lambdaPar,
        iterationsNumb,
        epsil,
        methodTV,
        nonneg,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int TV_FGP_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar,
        # int iter, float epsil, int methodTV, int nonneg, int gpu_device, int N, int M, int Z);
        cilregcuda.TV_FGP_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_int,  # iter (int)
            ctypes.c_float,  # epsil (float)
            ctypes.c_int,  # methodTV (int)
            ctypes.c_int,  # nonneg (int)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # Z (int)
        ]
        cilregcuda.TV_FGP_GPU_main.restype = ctypes.c_int  # return value is float

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
        if gpu_device == "gpu":
            gpu_device = 0
        # TV_FGP_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar,
        # int iter, float epsil, int methodTV, int nonneg, int gpu_device, int N, int M, int Z);
        cilregcuda.TV_FGP_GPU_main(
            in_p,
            out_p,
            infovector_p,
            lambdaPar,
            iterationsNumb,
            epsil,
            methodTV,
            nonneg,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out

    def PDTV_GPU(
        Input,
        lambdaPar,
        iter,
        epsil,
        lipschitz_const,
        methodTV,
        nonneg,
        gpu_device,
        Output=None,
        infovector=None,
    ):
        # int TV_PD_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iter, float epsil,
        # float lipschitz_const, int methodTV, int nonneg, int gpu_device, int dimX, int dimY, int dimZ);
        cilregcuda.TV_PD_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_int,  # iter (int)
            ctypes.c_float,  # epsil (float)
            ctypes.c_float,  # lipschitz_const (float)
            ctypes.c_int,  # methodTV (int)
            ctypes.c_int,  # nonneg (int)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # dimX (int)
            ctypes.c_int,  # dimY (int)
            ctypes.c_int,  # dimZ (int)
        ]
        cilregcuda.TV_PD_GPU_main.restype = ctypes.c_int  # return value is int

        input_p = Input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if Output is None:
            Output = np.zeros_like(Input)
        output_p = Output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if infovector is None:
            infovector = np.zeros_like(Input)
        infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        dims = list(Input.shape)[::-1]
        if Input.ndim == 2:
            dims.append(1)
        # int TV_PD_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iter, float epsil,
        # float lipschitz_const, int methodTV, int nonneg, int gpu_device, int dimX, int dimY, int dimZ);
        result = cilregcuda.TV_PD_GPU_main(
            input_p,
            output_p,
            infovector_p,
            lambdaPar,
            iter,
            epsil,
            lipschitz_const,
            methodTV,
            nonneg,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return Output

    def SB_TV_GPU(
        inputData,
        lambdaPar,
        iterationsNumb,
        epsil,
        methodTV,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int TV_SB_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar,
        # int iter, float epsil, int methodTV, int gpu_device, int N, int M, int Z);
        cilregcuda.TV_SB_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_int,  # iter (int)
            ctypes.c_float,  # epsil (float)
            ctypes.c_int,  # methodTV (int)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # Z (int)
        ]
        cilregcuda.TV_SB_GPU_main.restype = ctypes.c_int  # return value is int

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

        # int TV_SB_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iter, float epsil,
        # int methodTV, int gpu_device, int N, int M, int Z);
        result = cilregcuda.TV_SB_GPU_main(
            in_p,
            out_p,
            infovector_p,
            lambdaPar,
            iterationsNumb,
            epsil,
            methodTV,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out

    def LLT_ROF_GPU(
        inputData,
        lambdaROF,
        lambdaLLT,
        iterationsNumb,
        tau,
        epsil,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int LLT_ROF_GPU_main(float *Input, float *Output, float *infovector, float lambdaROF, float lambdaLLT,
        # int iterationsNumb, float tau,  float epsil, int gpu_device, int N, int M, int Z);
        cilregcuda.LLT_ROF_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaROF (float)
            ctypes.c_float,  # lambdaLLT (float)
            ctypes.c_int,  # iterationsNumb (int)
            ctypes.c_float,  # tau (float)
            ctypes.c_float,  # epsil (float)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # Z (int)
        ]
        cilregcuda.LLT_ROF_GPU_main.restype = ctypes.c_int  # return value is int

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

        # int LLT_ROF_GPU_main(float *Input, float *Output, float *infovector, float lambdaROF, float lambdaLLT,
        # int iterationsNumb, float tau,  float epsil, int gpu_device, int N, int M, int Z);
        result = cilregcuda.LLT_ROF_GPU_main(
            in_p,
            out_p,
            infovector_p,
            lambdaROF,
            lambdaLLT,
            iterationsNumb,
            tau,
            epsil,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out

    def TGV_GPU(
        inputData,
        lambdaPar,
        alpha1,
        alpha0,
        iterationsNumb,
        L2,
        epsil,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int TGV_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float alpha1, float alpha0,
        # int iterationsNumb, float L2, float epsil, int gpu_device, int dimX, int dimY, int dimZ);
        cilregcuda.TGV_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_float,  # alpha1 (float)
            ctypes.c_float,  # alpha0 (float)
            ctypes.c_int,  # iterationsNumb (int)
            ctypes.c_float,  # L2 (float)
            ctypes.c_float,  # epsil (float)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # dimX (int)
            ctypes.c_int,  # dimY (int)
            ctypes.c_int,  # dimZ (int)
        ]
        cilregcuda.TGV_GPU_main.restype = ctypes.c_int  # return value is int

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
        dims = dims[::-1]

        # int TGV_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float alpha1, float alpha0,
        # int iterationsNumb, float L2, float epsil, int gpu_device, int dimX, int dimY, int dimZ);
        result = cilregcuda.TGV_GPU_main(
            in_p,
            out_p,
            infovector_p,
            lambdaPar,
            alpha1,
            alpha0,
            iterationsNumb,
            L2,
            epsil,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out

    def dTV_FGP_GPU(
        inputData,
        inputRef,
        lambdaPar,
        iterationsNumb,
        epsil,
        eta,
        methodTV,
        nonneg,
        gpu_device,
        out=None,
        infovector=None,
    ):
        # int dTV_FGP_GPU_main(float *Input, float *InputRef, float *Output, float *infovector, float lambdaPar,
        # int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int gpu_device, int N, int M, int Z);
        cilregcuda.dTV_FGP_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_float),  # pointer to the InputRef array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Output array
            ctypes.POINTER(ctypes.c_float),  # pointer to the infovector array
            ctypes.c_float,  # lambdaPar (float)
            ctypes.c_int,  # iterationsNumb (int)
            ctypes.c_float,  # epsil (float)
            ctypes.c_float,  # eta (float)
            ctypes.c_int,  # methodTV (int)
            ctypes.c_int,  # nonneg (int)
            ctypes.c_int,  # gpu_device (int)
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # Z (int)
        ]
        cilregcuda.dTV_FGP_GPU_main.restype = ctypes.c_int  # return value is int

        in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inref_p = inputRef.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if out is None:
            out = np.zeros_like(inputData)
        out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if infovector is None:
            infovector = np.zeros((2,), dtype="float32")
        infovector_p = infovector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        dims = list(inputData.shape)[::-1]
        if inputData.ndim == 2:
            dims.append(1)

        # int dTV_FGP_GPU_main(float *Input, float *InputRef, float *Output, float *infovector, float lambdaPar,
        # int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int gpu_device, int N, int M, int Z);
        result = cilregcuda.dTV_FGP_GPU_main(
            in_p,
            inref_p,
            out_p,
            infovector_p,
            lambdaPar,
            iterationsNumb,
            epsil,
            eta,
            methodTV,
            nonneg,
            gpu_device,
            dims[0],
            dims[1],
            dims[2],
        )

        return out

    def PatchSelect_GPU(
        inputData,
        SearchWindow,
        SimilarWin,
        NumNeighb,
        h,
        gpu_device,
        H_i=None,
        H_j=None,
        H_k=None,
        Weights=None,
    ):
        # int PatchSelect_GPU_main(float *Input, unsigned short *H_i, unsigned short *H_j, float *Weights,
        # int N, int M, int SearchWindow, int SimilarWin, int NumNeighb, float h, int gpu_device);
        cilregcuda.PatchSelect_GPU_main.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to the Input array
            ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_i array
            ctypes.POINTER(ctypes.c_ushort),  # pointer to the H_j array
            ctypes.POINTER(ctypes.c_float),  # pointer to the Weights array
            ctypes.c_int,  # N (int)
            ctypes.c_int,  # M (int)
            ctypes.c_int,  # SearchWindow (int)
            ctypes.c_int,  # SimilarWin (int)
            ctypes.c_int,  # NumNeighb (int)
            ctypes.c_float,  # h (float)
            ctypes.c_int,  # gpu_device (int)
        ]
        cilregcuda.PatchSelect_GPU_main.restype = ctypes.c_int  # return value is int

        dims = [NumNeighb, inputData.shape[0], inputData.shape[1]]

        if Weights is None:
            Weights = np.zeros(dims, dtype="float32")
        if H_i is None:
            H_i = np.zeros(dims, dtype="uint16")
        if H_j is None:
            H_j = np.zeros(dims, dtype="uint16")

        in_p = inputData.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        hi_p = H_i.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        hj_p = H_j.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        weights_p = Weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # int PatchSelect_GPU_main(float *Input, unsigned short *H_i, unsigned short *H_j, float *Weights,
        # int N, int M, int SearchWindow, int SimilarWin, int NumNeighb, float h, int gpu_device);
        result = cilregcuda.PatchSelect_GPU_main(
            in_p,
            hj_p,
            hi_p,
            weights_p,
            dims[2],
            dims[1],
            SearchWindow,
            SimilarWin,
            NumNeighb,
            h,
            gpu_device,
        )

        return H_i, H_j, Weights
