"""Setup script for the DEODR project."""

import os
import re
from setuptools import setup, find_packages

from Cython.Build import cythonize

import numpy as np


extensions = "deodr/differentiable_renderer_cython.pyx"

my_modules = cythonize(extensions, annotate=True, language="c++")


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
    version=version,
    url="https://github.com/martinResearch/DEODR",
    data_files=[("C++", ["C++/DifferentiableRenderer.h"])],
    ext_modules=my_modules,  # additional source file(s)),
    include_dirs=[np.get_include()]    
)