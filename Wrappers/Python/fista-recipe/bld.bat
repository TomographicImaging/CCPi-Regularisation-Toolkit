IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\ccpifista"
xcopy /e "%RECIPE_DIR%\.." "%SRC_DIR%\ccpifista"

cd "%SRC_DIR%\ccpifista"
::%PYTHON% setup-fista.py -q bdist_egg
:: %PYTHON% setup.py install --single-version-externally-managed --record=record.txt
%PYTHON% setup-fista.py install
if errorlevel 1 exit 1
