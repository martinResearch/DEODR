from setuptools import setup
from Cython.Build import cythonize
import numpy as np

my_modules=cythonize("DEODR/differentiable_renderer_cython.pyx",annotate=True,language="c++")

libname="DEODR"
setup(
name = libname,
version="0.1",
author='Martin de La Gorce',
author_email='martin.delagorce@gmail.com',
license='BSD',
packages= ['DEODR'],
ext_modules = my_modules,  # additional source file(s)),
include_dirs=[ np.get_include()]
)


