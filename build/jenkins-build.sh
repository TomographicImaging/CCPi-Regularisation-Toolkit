#!/usr/bin/env bash
# Script to builds source code in Jenkins environment
module try-load conda

# install miniconda if the module is not present
if hash conda 2>/dev/null; then
  echo using conda
else
  if [ ! -f Miniconda3-latest-Linux-x86_64.sh ]; then
    wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
  fi
  ./Miniconda3-latest-Linux-x86_64.sh -u -b -p .
  PATH=$PATH:./bin
fi

# presume that git clone is done before this script launch
# git clone https://github.com/vais-ral/CCPi-Regularisation-Toolkit
conda install -y conda-build
#export CIL_VERSION=0.10.2
if [[ -n ${CIL_VERSION} ]]
then
  echo Using defined version: $CIL_VERSION
else
  export CIL_VERSION=0.10.3
  echo Defining version: $CIL_VERSION
fi
#cd CCPi-Regularisation-Toolkit # already there by jenkins
conda build Wrappers/Python/conda-recipe
