

from numpy import array as arr
from utilities import functions
from algorithms import trust_filter
from utilities import sys_utils
import program


# TODO:
# test with no constraints, only equality constraints, and only inequality constraints
# figure out why the algorithm in the paper doesn't work
# test on the dfovec methods
# test how to use quadratic constraints
# figure out how to decrease the trust region
# get another dfo method to work, and compare mine to theirs

sys_utils.clean_images_directory()

Q = arr([[1, 0],[0, 5]])
b = arr([0, 0])
c=0

q = functions.Quadratic(Q, b, c)
c1 = functions.Line(arr([1,  1]), -1)

equality = functions.VectorFunction(2)
equality.add(functions.Wiggles())

inequality = functions.VectorFunction(2)
inequality.add(functions.Line(arr([1, -1]), 10))

x0 = arr([1, 2])


program = program.Program("wiggles", q, equality, inequality, x0)

constants = trust_filter.Constants()
results = trust_filter.trust_filter(program, constants)

print(results)
print(results.x_min)
print(results.f_min)

