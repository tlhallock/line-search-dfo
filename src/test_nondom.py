
from nondom import NonDominatedSet as nondom
import numpy as np

s = nondom()

v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, 4])
v3 = np.array([3, 2, 1])


s.dominates([1, 2, 3])


