IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\test"
ROBOCOPY /E "%RECIPE_DIR%\..\test" "%SRC_DIR%\test"
cd %SRC_DIR%

:: issue cmake to create setup.py
cmake -G "NMake Makefiles" %RECIPE_DIR%\..\ -DBUILD_PYTHON_WRAPPERS=ON -DCONDA_BUILD=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB="%CONDA_PREFIX%\lib" -DLIBRARY_INC="%CONDA_PREFIX%" -DCMAKE_INSTALL_PREFIX="%PREFIX%\Library" 

nmake install
if errorlevel 1 exit 1
