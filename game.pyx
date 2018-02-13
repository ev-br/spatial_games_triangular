# distutils: language = c++

from libcpp.vector cimport vector
from numpy cimport import_array, PyArray_SimpleNewFromData, NPY_INT, npy_intp

cdef extern from "evolve.cc":
    void evolve_field(vector[int]&, int)



cdef class GameField:
    cdef vector[int] field
    cdef int L
    # XXX: p0, seed etc

    def __init__(self, int L):
        pass

    def __cinit__(self, int L):
        self.L = L
        self.field.resize(L*L)

        for i in range(L*L):
            self.field[i] = 0

    def get_field_array(self):
        """Return the field as a numpy array."""
        cdef npy_intp dims[2];
        dims[0] = self.L 
        dims[1] = self.L
        return PyArray_SimpleNewFromData(2, dims, NPY_INT, &self.field[0])

    def evolve(self, num_steps=1):
        evolve_field(self.field, num_steps)

#### init the numpy C API
import_array()
