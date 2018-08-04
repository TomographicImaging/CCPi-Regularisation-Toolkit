IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\ccpi"
ROBOCOPY /E "%RECIPE_DIR%\..\.." "%SRC_DIR%\ccpi"
ROBOCOPY /E "%RECIPE_DIR%\..\..\..\Core" "%SRC_DIR%\Core"
::cd %SRC_DIR%\ccpi\Python
cd %SRC_DIR%

:: issue cmake to create setup.py
cmake -G "NMake Makefiles" %RECIPE_DIR%\..\..\..\ -DBUILD_WRAPPERS=ON -DCONDA_BUILD=ON -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB="%CONDA_PREFIX%\lib" -DLIBRARY_INC="%CONDA_PREFIX%" -DCMAKE_INSTALL_PREFIX="%PREFIX%\Library"

::%PYTHON% setup-regularisers.py build_ext
::if errorlevel 1 exit 1
::%PYTHON% setup-regularisers.py install
::if errorlevel 1 exit 1
nmake install
if errorlevel 1 exit 1