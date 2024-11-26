try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
    
from Cython.Build import cythonize
import numpy

# TODO - Support GCC, this is for MSVC

setup(
    ext_modules=cythonize([Extension("dither.pattern.fast_candidates",
                                     sources=[r"dither\pattern\fast_candidates.pyx"],
                                     extra_compile_args=['/openmp', '/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()]),
                           Extension("dither.error_diffuse.lab_err_diff",
                                     sources=[r"dither\error_diffuse\lab_err_diff.pyx"],
                                     extra_compile_args=['/openmp', '/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()]),
                           Extension("dither.palette.k_means",
                                     sources=[r"dither\palette\k_means.pyx"],
                                     extra_compile_args=['/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()])],
                          compiler_directives={'language_level' : "3"})
)