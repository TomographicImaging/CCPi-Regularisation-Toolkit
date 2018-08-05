IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\build"
ROBOCOPY /E "%RECIPE_DIR%\..\..\Core" "%SRC_DIR%\build"
::ROBOCOPY /E "%RECIPE_DIR%\..\..\Wrappers\python\src" "%SRC_DIR%\build\module"
cd "%SRC_DIR%\build"

echo "we should be in %SRC_DIR%\build"

cmake -G "NMake Makefiles" "%RECIPE_DIR%\..\..\" -DLIBRARY_LIB="%CONDA_PREFIX%\lib" -DLIBRARY_INC="%CONDA_PREFIX%" -DCMAKE_INSTALL_PREFIX="%PREFIX%\Library" -DCONDA_BUILD=ON -DBUILD_WRAPPERS=OFF

::-DBOOST_LIBRARYDIR="%CONDA_PREFIX%\Library\lib" -DBOOST_INCLUDEDIR="%CONDA_PREFIX%\Library\include" -DBOOST_ROOT="%CONDA_PREFIX%\Library\lib"

:: Build C library
nmake install
if errorlevel 1 exit 1

:: Install step
