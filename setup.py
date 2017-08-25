from setuptools import setup


setup(name='ip-nonlinear-solver',
      version='0.1',
      description='Nonlinear programing solvers.',
      author='Antonio H. Ribeiro, Nikolay Mayorov, Matt Haberland, Ralf Gommers',
      author_email='antonior92@gmail.com',
      packages=['ipsolver'],
      test_suite='nose.collector',
      install_requires=['scipy>=0.19.1', 'numpy>=1.13.1'],
      test_require=['scipy>=0.19.1',
                    'numpy>=1.13.1',
                    'nose>=1.0',
                    'pytest'])
