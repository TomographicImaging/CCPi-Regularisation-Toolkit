IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\test"
ROBOCOPY /E "%RECIPE_DIR%\..\test" "%SRC_DIR%\test"

cmake -S "%SRC_DIR%" -B "%RECIPE_DIR%\.." -DBUILD_PYTHON_WRAPPERS=ON -DCONDA_BUILD=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE="RelWithDebInfo" -DLIBRARY_INC="%CONDA_PREFIX%" -DCMAKE_INSTALL_PREFIX="%SRC_DIR%\install"
cmake --build "%RECIPE_DIR%\.." --target install
%PYTHON% -m pip install "%SRC_DIR%\src\Python"

if errorlevel 1 exit 1
