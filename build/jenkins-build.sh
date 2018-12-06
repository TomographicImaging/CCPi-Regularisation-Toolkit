#!/usr/bin/env bash
# Script to builds source code in Jenkins environment

module avail
module load conda
# it expects that git clone is done before this script launch
# git clone https://github.com/vais-ral/CCPi-Regularisation-Toolkit
conda install -y conda-build
#export CIL_VERSION=0.10.2
export CIL_VERSION=0.10.2
#cd CCPi-Regularisation-Toolkit # already there by jenkins
conda build --debug Wrappers/Python/conda-recipe
