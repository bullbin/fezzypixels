from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# TODO - Support GCC, this is for MSVC

setup(
    ext_modules=cythonize([Extension("fezzypixels.pattern.fast_candidates",
                                     sources=[r"fezzypixels\pattern\fast_candidates.pyx"],
                                     extra_compile_args=['/openmp', '/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()]),
                           Extension("fezzypixels.error_diffuse.lab_err_diff",
                                     sources=[r"fezzypixels\error_diffuse\lab_err_diff.pyx"],
                                     extra_compile_args=['/openmp', '/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()]),
                           Extension("fezzypixels.palette.k_means",
                                     sources=[r"fezzypixels\palette\k_means.pyx"],
                                     extra_compile_args=['/Ox', '/fp:fast'],
                                     include_dirs=[numpy.get_include()])],
                          compiler_directives={'language_level' : "3"})
)