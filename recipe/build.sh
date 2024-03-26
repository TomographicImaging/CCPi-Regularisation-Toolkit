set -xe
cp -rv "$RECIPE_DIR/../test" "$SRC_DIR/"

cmake -S "$SRC_DIR" -B "$RECIPE_DIR/.." -DBUILD_PYTHON_WRAPPER=ON -DCONDA_BUILD=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE="RelWithDebInfo" -DLIBRARY_INC=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX="$SRC_DIR/install"
cmake --build "$RECIPE_DIR/.." --target install
$PYTHON -m pip install "$SRC_DIR/src/Python"
