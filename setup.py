from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

extensions = [
      Extension('game', ['game.pyx'], include_dirs = [np.get_include(), ], language="c++"),
      Extension('game', ['evolve.cc'], include_dirs = [np.get_include(), ], language="c++"),      
      ]


setup(ext_modules = cythonize(
           extensions,                          # our Cython source
           #include_path = [np.get_include()],   # does not work!!!
           #language="c++",                      # generate C++ code
      )
)