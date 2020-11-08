"""Setup script for the DEODR project."""

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


def get_version(filename):
    import os
    import re

    here = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(here, filename))
    version_file = f.read()
    f.close()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name=libname,
    version=get_version("deodr/__init__.py"),
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
    setup_requires=["numpy", "scipy", "cython"],
    install_requires=["numpy", "scipy"],
)
