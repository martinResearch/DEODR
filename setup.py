"""Setup script for the DEODR project."""

import os
import re
from setuptools import setup, find_packages

from Cython.Build import cythonize

import numpy as np


# compilation mode for debuging
# extensions = [
# Extension("differentiable_renderer_cython",
# ["DEODR/differentiable_renderer_cython.pyx"]
# ,extra_compile_args=["-Zi", "/Od"]
# ,extra_link_args=["-debug"],
# undef_macros = [ "NDEBUG" ]
# )
# ]

extensions = "deodr/differentiable_renderer_cython.pyx"

my_modules = cythonize(extensions, annotate=True, language="c++")

libname = "deodr"

with open(os.path.join(os.path.dirname(__file__), "deodr", "__init__.py")) as fp:
    for line in fp:
        m = re.search(r'^\s*__version__\s*=\s*([\'"])([^\'"]+)\1\s*$', line)
        if m:
            version = m.group(2)
            break
    else:
        raise RuntimeError("Unable to find own __version__ string")
print(f"version = {version}")
setup(
    name=libname,
    version=version,
    author="Martin de La Gorce",
    author_email="martin.delagorce@gmail.com",
    description="A differentiable renderer with Pytorch,Tensorflow and Matlab interfaces.",
    url="https://github.com/martinResearch/DEODR",
    license="BSD",
    packages=find_packages(),
    package_data={"deodr": ["*.pyx", "*.pxd", "data/*.*", "data/**/*.*"]},
    data_files=[("C++", ["C++/DifferentiableRenderer.h"])],
    ext_modules=my_modules,  # additional source file(s)),
    include_dirs=[np.get_include()],
    setup_requires=["numpy", "cython"],
    install_requires=["numpy", "scipy"],
)
