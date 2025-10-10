set -xe
cp -rv "$RECIPE_DIR/../test" "$SRC_DIR/"
if test -d "$SRC_DIR/build_proj"; then
  rm -rf "$SRC_DIR/build_proj"
fi

cmake -S "$SRC_DIR" -B "$SRC_DIR/build_proj" -G Ninja -DBUILD_PYTHON_WRAPPER=ON -DBUILD_CUDA=${BUILD_CUDA:-OFF} -DCMAKE_BUILD_TYPE="RelWithDebInfo" -DLIBRARY_INC="$CONDA_PREFIX" -DCMAKE_INSTALL_PREFIX="$SRC_DIR/install" ${CMAKE_ARGS}
cmake --build "$SRC_DIR/build_proj" --target install
$PYTHON -m pip install -vv "$SRC_DIR/src/Python"
