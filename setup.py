from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(
           ["game.pyx", "evolve.cc"],                 # our Cython source
#           sources=["evolve.cc"],  # additional source file(s)
           include_path = [np.get_include(),],
           language="c++",             # generate C++ code
      ))


