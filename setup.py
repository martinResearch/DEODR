from setuptools import setup,find_packages
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

#compilation mode for debuging
#extensions = [
    #Extension("differentiable_renderer_cython", ["DEODR/differentiable_renderer_cython.pyx"]
    #,extra_compile_args=["-Zi", "/Od"]
    #,extra_link_args=["-debug"],
	 #undef_macros = [ "NDEBUG" ] 
   #)    
#]

extensions="deodr/differentiable_renderer_cython.pyx"

my_modules=cythonize(extensions,annotate=True,language="c++")

libname="deodr"
setup(
name = libname,
version="0.1.3",
author='Martin de La Gorce',
author_email='martin.delagorce@gmail.com',
license='BSD',
packages= find_packages(),
package_data={'deodr': ['data/*.*','data/**/*.*']},
ext_modules = my_modules,  # additional source file(s)),
include_dirs=[ np.get_include()],
install_requires=['cython','numpy','scipy']
)

