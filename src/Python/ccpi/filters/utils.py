import platform
import ctypes
import os
import warnings

try:
    pre = {"Linux": "lib", "Windows": "", "Darwin": "lib"}[platform.system()]
except KeyError:
    raise ValueError(f"unsupported platform: {platform.system()}")
else:
    ext = {"Linux": ".so", "Windows": ".dll", "Darwin": ".dylib"}[platform.system()]

_here = os.path.dirname(__file__)
dll = f"{pre}cilreg{ext}"
cilreg = ctypes.cdll.LoadLibrary(os.path.join(_here, dll))

gpudll = f"{pre}cilregcuda{ext}"
try:
    cilregcuda = ctypes.cdll.LoadLibrary(os.path.join(_here, gpudll))
except Exception as exc:
    warnings.warn(str(exc), ImportWarning, stacklevel=2)
    warnings.warn(f"Found: {os.listdir(_here)}", ImportWarning, stacklevel=2)
    cilregcuda = None
