mkdir "%SRC_DIR%\test"
ROBOCOPY /E "%RECIPE_DIR%\..\test" "%SRC_DIR%\test"
if exist "%RECIPE_DIR%\..\build_proj" (
    rd /s /q "%RECIPE_DIR%\..\build_proj"
)
cmake -G "Visual Studio 16 2019" -S "%SRC_DIR%" -B "%RECIPE_DIR%\..\build_proj" -DBUILD_PYTHON_WRAPPERS=ON -DCONDA_BUILD=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE="RelWithDebInfo" -DLIBRARY_INC="%CONDA_PREFIX%" -DCMAKE_INSTALL_PREFIX="%RECIPE_DIR%\..\install"
cmake --build "%RECIPE_DIR%\..\build_proj" --target install --config RelWithDebInfo --verbose
%PYTHON% -m pip install "%SRC_DIR%\src\Python" --verbose

if errorlevel 1 exit 1
