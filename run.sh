#!/bin/bash  
echo "Building CCPi-regularisation Toolkit using CMake"  
# rm -r build
# Requires Cython, install it first: 
# pip install cython
# mkdir build
cd build/
make clean
# install Python modules only without CUDA
cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DBUILD_MATLAB_WRAPPER=OFF -DBUILD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
# install Python modules only with CUDA
# cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DBUILD_MATLAB_WRAPPER=OFF -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
# cp install/lib/libcilreg.so install/python/ccpi/filters
cd install/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
# spyder
# one can also run Matlab in Linux as:
# PATH="/path/to/mex/:$PATH" LD_LIBRARY_PATH="/path/to/library:$LD_LIBRARY_PATH" matlab
