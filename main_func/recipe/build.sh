#!/usr/bin/env bash

mkdir build
cd build

#configure
BUILD_CONFIG=Release
echo `pwd`
cmake .. -G "Ninja" \
      -Wno-dev \
      -DCMAKE_BUILD_TYPE=$BUILD_CONFIG \
      -DCMAKE_PREFIX_PATH:PATH="${PREFIX}" \
      -DCMAKE_INSTALL_PREFIX:PATH="${PREFIX}" \
      -DCMAKE_INSTALL_RPATH:PATH="${PREFIX}/lib"

# compile & install
ninja install
