
mkdir "$SRC_DIR/ccpi"
cp -rv "$RECIPE_DIR/../src/Matlab" "$SRC_DIR/ccpi"
cp -rv "$RECIPE_DIR/../src/Python" "$SRC_DIR/ccpi"
cp -rv "$RECIPE_DIR/../src/Core" "$SRC_DIR/Core"

cd $SRC_DIR
##cuda=off

cmake -G "Unix Makefiles" $RECIPE_DIR/../ -DBUILD_PYTHON_WRAPPER=ON -DCONDA_BUILD=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB=$CONDA_PREFIX/lib -DLIBRARY_INC=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$PREFIX


make install

#$PYTHON setup-regularisers.py build_ext
#$PYTHON setup-regularisers.py install


