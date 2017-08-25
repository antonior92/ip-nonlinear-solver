from setuptools import setup
from Cython.Build import cythonize

setup(name='ip-nonlinear-solver',
      version='0.1',
      description='A trust-region interior-point method for general nonlinear programing problems.',
      author='Antonio Horta Ribeiro',
      author_email='antonior92@gmail.com',
      ext_modules=cythonize('ipsolver/_group_columns.pyx'),
      packages=['ipsolver'])
