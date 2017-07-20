

from oct2py import octave
import os
import numpy as np
import utilities.cute_dfo_functions as funs


octave.addpath(os.path.join(os.path.dirname(os.path.realpath(__file__))), '../octave')

x = np.random.random(3)
res = octave.dfovec(2, 2, x, 1)

print(res)

res = funs.dfovec(2, 2, x, 1)

print(res)
