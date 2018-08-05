
if [ -z "$CIL_VERSION" ]; then
    echo "Need to set CIL_VERSION"
    exit 1
fi  
mkdir "$SRC_DIR/ccpi"
cp -rv "$RECIPE_DIR/../.." "$SRC_DIR/ccpi"
cp -rv "$RECIPE_DIR/../../../Core" "$SRC_DIR/Core"

cd $SRC_DIR

cmake -G "NMake Makefiles" $RECIPE_DIR/../../../ -DBUILD_WRAPPERS=ON -DCONDA_BUILD=ON -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB=$CONDA_PREFIX/lib -DLIBRARY_INC=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$PREFIX/Library"


make install

#$PYTHON setup-regularisers.py build_ext
#$PYTHON setup-regularisers.py install


