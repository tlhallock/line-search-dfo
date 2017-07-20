
from numpy import array
from utilities import functions
from utilities import create_program
import matplotlib.pyplot as plt


quad1 = functions.Quadratic(array([[1, 0], [0, 2]]), array([0, 0]), 5)
quad2 = functions.Quadratic(array([[-1, 0], [0, -2]]), array([10, -1]), -13)
quad3 = functions.Quadratic(array([[-1, 0], [0, -2]]), array([-1, 1]), 50)

func = lambda x: array([quad1.evaluate(x), quad2.evaluate(x), quad3.evaluate(x)])



program, center = create_program.create_program('test', func, 2, 3, num_equality=1)

program.createBasePlotAt(center, 20)
plt.show()



#
#
# from oct2py import octave
# import os
# import numpy as np
# import utilities.cute_dfo_functions as funs
#
# octave.addpath(os.path.dirname(os.path.realpath(__file__)))
#
#
# x = np.random.random(3)
# res = octave.dfovec(2, 2, x, 1)
#
# print(res)
#
# res = funs.dfovec(2, 2, x, 1)
#
# print(res)

