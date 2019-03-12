#!/bin/bash  
echo "Building CCPi-regularisation Toolkit using CMake"  
rm -r build_proj
# Requires Cython, install it first: 
# pip install cython
mkdir build_proj
cd build_proj/
#make clean
export CIL_VERSION=19.03
# install Python modules without CUDA
# cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DBUILD_MATLAB_WRAPPER=OFF -DBUILD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
# install Python modules with CUDA
# cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DBUILD_MATLAB_WRAPPER=OFF -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
# install Matlab modules without CUDA
#cmake ../ -DBUILD_PYTHON_WRAPPER=OFF -DMatlab_ROOT_DIR=/dls_sw/apps/matlab/r2014a/ -DBUILD_MATLAB_WRAPPER=ON -DBUILD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
# install Matlab modules with CUDA
cmake ../ -DBUILD_PYTHON_WRAPPER=OFF -DMatlab_ROOT_DIR=/dls_sw/apps/matlab/r2014a/ -DBUILD_MATLAB_WRAPPER=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
#### Python
#cp install/lib/libcilreg.so install/python/ccpi/filters
# cd install/python
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
# spyder
##### Matlab (Linux)
#PATH="/path/to/mex/:$PATH" LD_LIBRARY_PATH="/path/to/library:$LD_LIBRARY_PATH" matlab
PATH="/home/kjy41806/Documents/SOFT/CCPi-Regularisation-Toolkit/build_proj/install/matlab/:$PATH" LD_LIBRARY_PATH="/home/kjy41806/Documents/SOFT/CCPi-Regularisation-Toolkit/build_proj/install/lib:$LD_LIBRARY_PATH" matlab
