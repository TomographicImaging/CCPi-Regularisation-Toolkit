#!/usr/bin/env bash
# TODO put git tag recognition
if [[ -n ${CIL_VERSION} ]]
then
  echo Using defined version: $CIL_VERSION
else
# TODO put git tag recognition, or default
  #export CIL_VERSION=0.10.4
  #get tag, remove first char ('v') and leave rest
  export CIL_VERSION=`git describe --tags | tail -c +2`  
  if [[ ${CIL_VERSION} == *"-"* ]]; then
    # detected dash means that it is dev version
    # version is then string-string and all after second dash is ignored (usually commit sha)
    export CIL_VERSION=`echo ${CIL_VERSION} | cut -d "-" -f -2`
    # but dash is prohibited for conda build
    export CIL_VERSION=`echo ${CIL_VERSION} | tr - _`
    echo Building dev version ${CIL_VERSION}
  else
    echo Defining version from last git tag and commit: $CIL_VERSION
  fi 
fi
# Script to builds source code in Jenkins environment
# module try-load conda

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

# presume that git clone is done before this script is launched, if not, uncomment
#git clone https://github.com/vais-ral/CCPi-Regularisation-Toolkit
conda install -y conda-build
#export CIL_VERSION=0.10.2
#cd CCPi-Regularisation-Toolkit # already there by jenkins
# need to call first build
conda build Wrappers/Python/conda-recipe
# then need to call the same with --output 
#- otherwise no build is done :-(, just fake file names are generated
export REG_FILES=`conda build Wrappers/Python/conda-recipe --output`
# REG_FILES variable should contain output files
echo files created: $REG_FILES
#upload to anaconda
if [[ -n ${CCPI_CONDA_TOKEN} ]]
then
  conda install anaconda-client
  while read -r outfile; do
    #TODO if git tag is defined than call anaconda without --label dev
    #TODO if pull request??? do not upload 
    if [[ $CIL_VERSION == *"_"*]]; then
      # upload with dev label
      anaconda -v -t ${CCPI_CONDA_TOKEN}  upload $outfile --force --label dev
    else 
      anaconda -v -t ${CCPI_CONDA_TOKEN}  upload $outfile --force
    fi
  done <<< "$REG_FILES"
else
  echo CCPI_CONDA_TOKEN not defined, will not upload to anaconda.
fi
