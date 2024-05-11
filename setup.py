"""Setup script for the DEODR project."""

import os
import re

from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, find_packages, setup

extension = Extension(
    name="deodr.differentiable_renderer_cython",
    sources=["deodr/differentiable_renderer_cython.pyx"],
    include_dirs=["deodr", np.get_include()],
)

with open(os.path.join(os.path.dirname(__file__), "deodr", "__init__.py")) as fp:
    for line in fp:
        m = re.search(r'^\s*__version__\s*=\s*([\'"])([^\'"]+)\1\s*$', line)
        if m:
            version = m.group(2)
            break
    else:
        raise RuntimeError("Unable to find own __version__ string")
print(f"version = {version}")

readme_file = os.path.join(os.path.dirname(__file__), "readme.md")
setup(
    version=version,
    url="https://github.com/martinResearch/DEODR",
    data_files=[("C++", ["C++/DifferentiableRenderer.h"])],
    ext_modules=cythonize([extension], annotate=True, build_dir="build"),
    include_dirs=[np.get_include()],
    packages=find_packages(),
    package_data={"deodr": ["*.pyx", "*.pxd", "data/*.*", "data/**/*.*"]},
    long_description=open(readme_file, "r").read(),
    long_description_content_type="text/markdown",
)
