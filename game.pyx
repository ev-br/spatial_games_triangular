# distutils: language=c++
import numpy as np
import matplotlib.pyplot as plt

from libcpp.vector cimport vector
from numpy cimport import_array, PyArray_SimpleNewFromData, NPY_INT, npy_intp

cdef extern from "evolve.cc":
    void evolve_field(vector[int]&, double, int) nogil

cdef class GameField:
    cdef vector[int] _field
    cdef int L
    cdef double b

    def __init__(self, L, b, per, seed=0):
        pass

#    def __cinit__(self, int L, double b):
#        self.L = L
#        self.b = b
#        self._field.resize(L*L)
#
#        for i in range(L*L):
#            self._field[i] = 0

    def __cinit__(self, int L, double b, double per, int seed=0):
        self.L = L
        self.b = b
        self._field.resize(L*L)

        if seed == -1:
            self._field = np.array([int(i == L*L // 2) for i in range(L*L)])
        else:
            if not seed == 0:
                np.random.seed(seed)
                
            for i in range(L*L):
                self._field[i] = int(100 * np.random.rand() < per)

    @property
    def field(self):
        """Return the field as a numpy array."""
        cdef npy_intp dims[2]
        dims[0] = self.L 
        dims[1] = self.L
        return PyArray_SimpleNewFromData(2, dims, NPY_INT, &self._field[0])

    @property
    def size(self):
        """Return field size."""
        return self.L

    @field.setter
    def field(self, arr):
        """Set the game field."""
        arr = np.asarray(arr)
        if len(arr.shape) != 2:
            raise ValueError("Expected a 2D array, got %s-d." % len(arr.shape))
        if arr.size != self.L*self.L:
            raise ValueError("Size mismatch: expected %s, got %s." % (self.L*self.L, arr.size))

        arr = arr.ravel()
        for j in range(arr.size):
            self._field[j] = arr[j]

    def evolve(self, int num_steps=1):
        with nogil:
            evolve_field(self._field, self.b, num_steps)

    def show(self, point_size=10, scale=1):
        plt.clf()
        plt.figure(figsize = (scale*10, scale*6.66))
        y, x = (1-self.field).nonzero()
        plt.scatter(x + y*np.sin(np.pi/6), y * np.sin(np.pi/3), s=point_size, marker='h')
        y, x = self.field.nonzero()
        plt.scatter(x + y*np.sin(np.pi/6), y * np.sin(np.pi/3), s=point_size, marker='h', c='r')

#### init the numpy C API
import_array()
