import platform
import ctypes

if platform.system() == 'Linux':
    dll = 'libcilreg.so'
elif platform.system() == 'Windows':
    dll_file = 'cilreg.dll'
    dll = ctypes.util.find_library(dll_file)
elif platform.system() == 'Darwin':
    dll = 'libcilreg.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

cilreg = ctypes.cdll.LoadLibrary(dll)

try:
    if platform.system() == 'Linux':
        gpudll = 'libcilregcuda.so'
    elif platform.system() == 'Windows':
        gpudll_file = 'cilregcuda.dll'
        gpudll = ctypes.util.find_library(gpudll_file)
    elif platform.system() == 'Darwin':
        gpudll = 'libcilregcuda.dylib'
    else:
        raise ValueError('Not supported platform, ', platform.system())
    
    cilregcuda = ctypes.cdll.LoadLibrary(gpudll)
except OSError as ose:
    print(ose)
    cilregcuda = None