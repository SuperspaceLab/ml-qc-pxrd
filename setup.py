from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

# python setup.py build_ext --inplace
ext = Extension("pxrd", sources=["generator.pyx"], include_dirs=['.', get_include()])
setup(name="pxrd", ext_modules=cythonize([ext]))
