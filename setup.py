from setuptools import setup
from Cython.Build import cythonize
import numpy

ext_module = cythonize("ipsolver/_group_columns.pyx")
ext_module[0].include_dirs = [numpy.get_include(), '.']

setup(name='ip-nonlinear-solver',
      version='0.1',
      description='A trust-region interior-point method for general nonlinear programing problems.',
      author='Antonio Horta Ribeiro',
      author_email='antonior92@gmail.com',
      ext_modules=ext_module,
      packages=['ipsolver'])
