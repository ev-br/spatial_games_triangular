#
# https://matplotlib.org/examples/animation/dynamic_image.html
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from game import GameField

# prepare the field
L, b = 4, 1.81
f = GameField(L, b)
a = np.zeros((L, L), dtype=int)
a[::2, ::2] = 1
a[1::2, 1::2] = 1
f.field = a

# draw the initial field
fig = plt.figure()
im = plt.imshow(a, animated=True)

# updater function
def updatefig(*args):
    f.evolve()
    im.set_array(f.field)
    return im,

# animate!
ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()

