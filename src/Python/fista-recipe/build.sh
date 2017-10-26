if [ -z "$CIL_VERSION" ]; then
    echo "Need to set CIL_VERSION"
    exit 1
fi  
mkdir "$SRC_DIR/ccpifista"
cp -r "$RECIPE_DIR/.." "$SRC_DIR/ccpifista"

cd $SRC_DIR/ccpifista

$PYTHON setup-fista.py install
