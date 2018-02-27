import numpy as np
from numpy.testing import assert_equal, assert_

from game import GameField

def test_setter():
    a = np.ones((4, 4), dtype=int)
    f = GameField(4, b=1.81)
    f.field = a

    assert_equal(f.field, a)


def test_evolves():
    a = np.zeros((4, 4), dtype=int)
    a[::2, ::2] = 1
    a[1::2, 1::2] = 1
    f = GameField(4, b=1.81)
    f.field = a
    f.evolve()
    assert (f.field != a).all()


def test_num_steps():
    a = np.zeros((4, 4), dtype=int)
    a[::2, ::2] = 1
    a[1::2, 1::2] = 1

    f = GameField(4, b=1.81)
    f.field = a
    f.evolve(num_steps=2)

    f1 = GameField(4, b=1.81)
    f1.field = a
    f1.evolve()
    f1.evolve()

    assert_equal(f.field, f1.field)
