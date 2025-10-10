mkdir "%SRC_DIR%\test"
ROBOCOPY /E "%RECIPE_DIR%\..\test" "%SRC_DIR%\test"
if exist "%SRC_DIR%\build_proj" (
    rd /s /q "%SRC_DIR%\build_proj"
)

:: -G "Visual Studio 16 2019" specifies the the generator
:: -T v142 specifies the toolset
:: -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" to the cmake command to specify the CUDA toolkit version

cmake -S "%SRC_DIR%" -B "%SRC_DIR%\build_proj" -G Ninja -DBUILD_PYTHON_WRAPPERS=ON -DBUILD_CUDA=%BUILD_CUDA% -DCMAKE_BUILD_TYPE="RelWithDebInfo" -DLIBRARY_INC="%CONDA_PREFIX%" -DCMAKE_INSTALL_PREFIX="%SRC_DIR%\install" %CMAKE_ARGS%
cmake --build "%SRC_DIR%\build_proj" --target install --config RelWithDebInfo
%PYTHON% -m pip install -vv "%SRC_DIR%\src\Python"

if errorlevel 1 exit 1
