IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\ccpi"
ROBOCOPY /E "%RECIPE_DIR%\..\.." "%SRC_DIR%\ccpi"
ROBOCOPY /E "%RECIPE_DIR%\..\..\..\Core" "%SRC_DIR%\Core"
cd %SRC_DIR%\ccpi\Python

:: issue cmake to create setup.py
cmake . 

%PYTHON% setup-regularizers.py build_ext
if errorlevel 1 exit 1
%PYTHON% setup-regularizers.py install
if errorlevel 1 exit 1
