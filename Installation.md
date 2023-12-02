## Installation:

In order to compile C/C++ sources and additional wrappers from source code for numpy 1.24 and python 3.10, the recommended way is:
```
git clone https://github.com/vais-ral/CCPi-Regularisation-Toolkit
cd CCPi-Regularisation-Toolkit
export CCPI_BUILD_ARGS="--numpy 1.24 --python 3.10"
build/jenkins-build.sh
```
this will install `conda build` environment and compiles C/C++ and Python wrappers and performs basic tests for environment with python 3.10 and numpy 1.24.

### CMake
If you want to build directly using cmake, install CMake (v.>=3) to configure it. Additionally you will need a C compiler, `make` (on linux) and CUDA SDK where available. The toolkit may be used directly from C/C++ as it is compiled as a shared library (check-out the include files in `Core` for this)
1. Clone this repository to a directory, i.e. `CCPi-Regularisation-Toolkit`,
2. create a build directory.
3. Issue `cmake` to configure (or `cmake-gui`, or `ccmake`, or `cmake3`). Use additional flags to fine tune the configuration.

Flags used during configuration

| CMake flag | type | meaning |
|:---|:----|:----|
| `BUILD_PYTHON_WRAPPER` | bool | `ON\|OFF` whether to build the Python wrapper |
| `BUILD_MATLAB_WRAPPER` | bool | `ON\|OFF` whether to build the Matlab wrapper |
| `CMAKE_INSTALL_PREFIX` | path | your favourite install directory |
| `PYTHON_DEST_DIR` | path | python modules install directory (default `${CMAKE_INSTALL_PREFIX}/python`) |
| `MATLAB_DEST_DIR` | path | Matlab modules install directory (default `${CMAKE_INSTALL_PREFIX}/matlab`)|
| `BUILD_CUDA` | bool | `ON\|OFF` whether to build the CUDA regularisers |
| `CONDA_BUILD`| bool | `ON\|OFF` whether it is installed with `setup.py install`|
| `Matlab_ROOT_DIR` | path | Matlab directory|
|`PYTHON_EXECUTABLE` | path | /path/to/python/executable|

Here an example of build on Linux (see also `run.sh` for additional info):

```bash
git clone https://github.com/vais-ral/CCPi-Regularisation-Toolkit.git
cd build
cmake .. -DCONDA_BUILD=OFF -DBUILD_MATLAB_WRAPPER=ON -DBUILD_PYTHON_WRAPPER=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
cd install/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
```

### Python
#### Python binaries
Python binaries are distributed via the [ccpi](https://anaconda.org/ccpi/ccpi-regulariser) conda channel.

```
conda install ccpi-regulariser -c ccpi -c conda-forge
```

#### Python (conda-build)
```
export CIL_VERSION=$(date +%Y.%m) (Unix) / set CIL_VERSION=19.10 (Windows)
conda build recipe/ --numpy 1.23 --python 3.10
conda install ccpi-regulariser=${CIL_VERSION} --use-local --force-reinstall # doesn't work?
conda install -c file://${CONDA_PREFIX}/conda-bld/ ccpi-regulariser=${CIL_VERSION} --force-reinstall # try this one
cd demos/
python demo_cpu_regularisers.py # to run CPU demo
python demo_gpu_regularisers.py # to run GPU demo
```

### Python (GPU-CuPy)
One can also use some of the GPU modules directly (i.e. without the need of building the package) by using [CuPy](https://docs.cupy.dev/en/stable/index.html) implementations. 
```
git clone git@github.com:vais-ral/CCPi-Regularisation-Toolkit.git
cd CCPi-Regularisation-Toolkit
pip install .
``` 

#### Python build
If passed `CONDA_BUILD=ON` the software will be installed by issuing `python setup.py install` which will install in the system python (or whichever other python it's been picked up by CMake at configuration time.)
If passed `CONDA_BUILD=OFF` the software will be installed in the directory pointed by `${PYTHON_DEST_DIR}` which defaults to `${CMAKE_INSTALL_PREFIX}/python`. Therefore this directory should be added to the `PYTHONPATH`.

If Python is not picked by CMake you can provide the additional flag to CMake `-DPYTHON_EXECUTABLE=/path/to/python/executable`.

### MultiGPU capability (to use in Python with mpi4py)
The toolkit can be used by running in parallel across multiple GPU devices on a PC or a compute node of a cluster. In order to initiate a parallel run on your GPUs you will need an MPI library, such as, [mpi4py](https://mpi4py.readthedocs.io/en/stable/). See multi_gpu demo script in demos folder, it can be run as
```
mpirun -np 2 python multi_gpu.py -g -s -gpus 2
```
where `-np` parameter defines the total number of processes and `-gpus` defines the number of available GPUs. 

### Matlab

Matlab wrapper will install in the `${MATLAB_DEST_DIR}` directory, which defaults to `${CMAKE_INSTALL_PREFIX}/matlab`

If Matlab is not picked by CMake, you could add `-DMatlab_ROOT_DIR=<Matlab directory>`.

#### Linux
Because you've installed the modules in `<your favourite install directory>` you need to instruct Matlab to look in those directories:

```bash

PATH="/path/to/mex/:$PATH" LD_LIBRARY_PATH="/path/to/library:$LD_LIBRARY_PATH" matlab
```
By default `/path/to/mex` is `${CMAKE_INSTALL_PREFIX}/bin` and `/path/to/library/` is `${CMAKE_INSTALL_PREFIX}/lib`

#### Windows
On Windows the `dll` and the mex modules must reside in the same directory. It is sufficient to add the directory at the beginning of the m-file.
```matlab
addpath(/path/to/library);
```

#### Legacy Matlab installation (partly supported, please use Cmake)
```
	cd src/Matlab/mex_compile
	compileCPU_mex.m % to compile CPU modules
	compileGPU_mex.m % to compile GPU modules (see instructions in the file)
```