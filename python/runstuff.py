

from numpy import array as arr



from utilities import functions

import scipy

Q = arr([[1, 0],[0, 5]])
b = arr([0, 0])
c=0

q = functions.Quadratic(Q, b, c)
c1 = functions.Line(arr([1,  1]), -1)

equality = [functions.Wiggles()]

inequality = [functions.Line(arr([1, -1]), 10)]

x0 = arr([1, 2])


scipy.minimize(lambda x: q.evaluate(x))

