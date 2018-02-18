# distutils: language = c++
import numpy as np

from libcpp.vector cimport vector
from numpy cimport import_array, PyArray_SimpleNewFromData, NPY_INT, npy_intp

cdef extern from "evolve.cc":
    void evolve_field(vector[int]&, double, int) nogil
    void fake_evolve(vector[int]&, double, int) nogil


cdef class GameField:
    cdef vector[int] _field
    cdef int L
    cdef double b

    def __init__(self, L, b):
        pass

    def __cinit__(self, int L, double b):
        self.L = L
        self.b = b
        self._field.resize(L*L)

        for i in range(L*L):
            self._field[i] = 0

    @property
    def field(self):
        """Return the field as a numpy array."""
        cdef npy_intp dims[2]
        dims[0] = self.L 
        dims[1] = self.L
        return PyArray_SimpleNewFromData(2, dims, NPY_INT, &self._field[0])

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
            fake_evolve(self._field, self.b, num_steps)

#### init the numpy C API
import_array()
