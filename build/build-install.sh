#!/usr/bin/env bash
export CCPI_BUILD_ARGS="--numpy 1.12 --python 3.6"
bash <(curl -L https://raw.githubusercontent.com/vais-ral/CCPi-VirtualMachine/master/scripts/jenkins-build.sh)
conda install -y ccpi-regulariser --use-local --force