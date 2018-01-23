
if [ -z "$CIL_VERSION" ]; then
    echo "Need to set CIL_VERSION"
    exit 1
fi  
mkdir "$SRC_DIR/ccpi"
cp -rv "$RECIPE_DIR/../.." "$SRC_DIR/ccpi"
cp -rv "$RECIPE_DIR/../../../Core" "$SRC_DIR/Core"

cd $SRC_DIR/ccpi/Python

echo "$SRC_DIR/ccpi/Python"

$PYTHON setup.py build_ext
$PYTHON setup.py install


