
from utilities import functions

from numpy.random import permutation
from numpy.random import random
from numpy import arange
from numpy import ones
from numpy import zeros
import program

def create_program(name, func, indim, outdim, num_equality=None, percentage_equality=.5):
	if num_equality is None:
		num_equality = min(indim, int(percentage_equality * (outdim - 1)))

	level_set_height = 5
	radius = 50

	perm = permutation(arange(outdim))

	x0 = (radius / 2) * ones(indim) - radius * random(indim)
	curval = func(x0)

	class obj:
		def __init__(self, i):
			self.i = i
		def evaluate(self, x):
			return func(x)[self.i] - curval[self.i]

	if num_equality > 0:
		equality_constraints = functions.VectorFunction(indim)
		for i in range(1, 1 + num_equality):
			equality_constraints.add(obj(perm[i]))
	else:
		equality_constraints = None

	num_inequality = outdim - 1 - num_equality
	if num_inequality > 0:
		inequality_constraints = functions.VectorFunction(indim)
		for i in range(1 + num_equality, outdim):
			inequality_constraints.add(obj(perm[i]))
	else:
		inequality_constraints = None

	return program.Program(name, obj(perm[0]), equality_constraints, inequality_constraints, zeros(indim)), x0
