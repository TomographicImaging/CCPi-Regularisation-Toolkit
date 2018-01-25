#!/usr/bin/env bash

echo $CIL_VERSION
#if [ -z "$CIL_VERSION" ]; then
#    echo "Need to set CIL_VERSION"
#    exit 1
#fi  
#export CIL_VERSION=0.9.1



mkdir ${SRC_DIR}/build
cp -rv ${RECIPE_DIR}/../../Core/ ${SRC_DIR}/build
mkdir ${SRC_DIR}/build/build
cd ${SRC_DIR}/build/build
cmake -G "Unix Makefiles" -DLIBRARY_LIB="${CONDA_PREFIX}/lib" -DLIBRARY_INC="${CONDA_PREFIX}" -DCMAKE_INSTALL_PREFIX="${PREFIX}" ../Core

make -j2 VERBOSE=1
make install
