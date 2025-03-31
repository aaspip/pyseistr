#!/usr/bin/env python
# -*- encoding: utf8 -*-
import io
import os

from setuptools import setup
from distutils.core import Extension
import numpy


long_description = """
Source code: https://github.com/aaspip/pyseistr""".strip() 


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")).read()

dipc_module = Extension('dipcfun', sources=['pyseistr/src/dip_cfuns.c'], 
										include_dirs=[numpy.get_include()])
sofc_module = Extension('sofcfun', sources=['pyseistr/src/sof_cfuns.c'], 
										include_dirs=[numpy.get_include()])
sofc3d_module = Extension('sof3dcfun', sources=['pyseistr/src/sof3d_cfuns.c'], 
										include_dirs=[numpy.get_include()])
sointc3d_module = Extension('soint3dcfun', sources=['pyseistr/src/soint3d_cfuns.c'], 
										include_dirs=[numpy.get_include()])
sointc2d_module = Extension('soint2dcfun', sources=['pyseistr/src/soint2d_cfuns.c'], 
										include_dirs=[numpy.get_include()])
bpc_module = Extension('bpcfun', sources=['pyseistr/src/bp_cfuns.c'], 
										include_dirs=[numpy.get_include()])
cohc_module = Extension('cohcfun', sources=['pyseistr/src/coh_cfuns.c'], 
										include_dirs=[numpy.get_include()])
paintc2d_module = Extension('paint2dcfun', sources=['pyseistr/src/paint_cfuns.c'], 
										include_dirs=[numpy.get_include()])
setup(
    name="pyseistr",
    version="0.0.4.4.2",
    license='MIT License',
    description="A python package for structural denoising and interpolation of multi-channel seismic data",
    long_description=long_description,
    author="pyseistr developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/aaspip/pyseistr",
    ext_modules=[dipc_module,sofc_module,sofc3d_module,sointc2d_module,sointc3d_module,bpc_module,paintc2d_module,cohc_module],
    packages=['pyseistr'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords=[
        "seismology", "earthquake seismology", "exploration seismology", "array seismology", "denoising", "science", "engineering", "structure", "local slope", "filtering"
    ],
    install_requires=[
        "numpy", "scipy", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
