from distutils.core import setup
#from setuptools import setup, find_packages
import os

cil_version=os.environ['CIL_VERSION']
if  cil_version == '':
    print("Please set the environmental variable CIL_VERSION")
    sys.exit(1)

setup(
    name="ccpi-fista",
    version=cil_version,
    packages=['ccpi','ccpi.reconstruction'],
	install_requires=['numpy'],

     zip_safe = False,

    # metadata for upload to PyPI
    author="Edoardo Pasca",
    author_email="edo.paskino@gmail.com",
    description='CCPi Core Imaging Library - FISTA Reconstructor module',
    license="Apache v2.0",
    keywords="tomography interative reconstruction",
    url="http://www.ccpi.ac.uk",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
