

import os
from program import Program
from numpy import array_equal
from numpy import asarray
from numpy import zeros
from utilities.sys_utils import createObject
from algorithms import trust_filter
from utilities import functions
from octave.pyfovec import dfovec
from octave.pydfox import dfoxs

class Context:
	def __init__(self):
		self.nfev = 0
		self.fvals = []  # This is a list now, need to change this for create_perf_plot


n = 2
m = 2
# Freudenstein and Roth function.
nprob = 7

context = Context()

# Caches a repeated call at the same value
class FunctionEvaluationCacher:
	def __init__(self, orig):
		self.orig = orig
		self.cacheX = None
		self.cacheY = None

	def evaluate(self, x):
		if array_equal(self.cacheX, x):
			return self.cacheY
		else:
			# Not thread safe...
			self.cacheX = x
			self.cacheY = self.orig(self.cacheX)
			return self.cacheY

# Takes the column matrix returned by dfovec and converts it into an array for the rest of the program
# toVector = lambda x: asarray(x).flatten().T

cacher = FunctionEvaluationCacher(lambda x: dfovec(m, n, x, nprob))
# an object with an evaluate method
objective = createObject()
objective.evaluate = lambda x: cacher.evaluate(x)[0]
objective.getOutDim = lambda: 1
objective.getFunction = lambda idx: objective
# an object with an evaluate method
dFovecInequality = createObject()
dFovecInequality.evaluate = lambda x: cacher.evaluate(x)[1]
dFovecInequality.getOutDim = lambda: 1
dFovecInequality.getFunction = lambda idx: dFovecInequality

# One constraint an then constrain it to a diamond
# This is just so that the linear program becomes bounded...
inequalities = functions.VectorFunction(n)
inequalities.add(dFovecInequality)
inequalities.add(functions.Line(asarray([1, 1]), -1))
inequalities.add(functions.Line(asarray([1, -1]), -1))
inequalities.add(functions.Line(asarray([-1, 1]), -1))
inequalities.add(functions.Line(asarray([-1, -1]), -1))


dfovecProgram = Program('dfovec',
			objective,
			None,
			inequalities,
			dfoxs(n, nprob, 1))


constants = trust_filter.Constants()
results = trust_filter.trust_filter(dfovecProgram, constants, plot=True)


print(results)
print(results.x_min)
print(results.f_min)

print(context.fvals)
print(context.nfev)
