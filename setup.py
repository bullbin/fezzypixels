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
                                     include_dirs=[numpy.get_include()])],
                          compiler_directives={'language_level' : "3"})
)