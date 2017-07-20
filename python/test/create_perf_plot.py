

from oct2py import octave
import os
from program import Program
from numpy import array_equal
from numpy import asarray
from utilities.sys_utils import createObject
from algorithms import trust_filter
from utilities import functions

def getOctavePath():
	curfile = os.path.realpath(__file__)
	curdir = os.path.dirname(curfile)
	octavedir = os.path.join(curdir, '../octave')
	return octavedir

octave.addpath(getOctavePath())

n = 2
m = 2
# Freudenstein and Roth function.
nprob = 7

# Are really only needed for calfun.m, which we cannot use...
octave.eval('global m = ' + str(m))
octave.eval('global nprob = ' + str(nprob))
octave.eval("global probtype = 'smooth'")
octave.eval('global fvals = zeros(1, ' + str(m) + ')')
octave.eval('global nfev = 1')
octave.eval('global np = 1')

# Caches a repeated call at the same value
class FunctionEvaluationCacher:
	def __init__(self, orig):
		self.orig = orig
		self.cacheX = None
		self.cacheY = None
	def  evaluate(self, x):
		if array_equal(self.cacheX, x):
			return self.cacheY
		else:
			self.cacheX = x
			self.cacheY = self.orig(self.cacheX)
			return self.cacheY

# Takes the column matrix returned by dfovec and converts it into an array for the rest of the program
toVector = lambda x: asarray(x).flatten().T

cacher = FunctionEvaluationCacher(lambda x: octave.dfovec(m, n, x, nprob))
# an object with an evaluate method
objective = createObject()
objective.evaluate = lambda x: toVector(cacher.evaluate(x))[0]
objective.getOutDim = lambda: 1
objective.getFunction = lambda idx: objective
# an object with an evaluate method
dFovecInequality = createObject()
dFovecInequality.evaluate = lambda x: toVector(cacher.evaluate(x))[1]
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
			toVector(octave.dfoxs(n, nprob, 1)))


constants = trust_filter.Constants()
results = trust_filter.trust_filter(dfovecProgram, constants, plot=False)


print(octave.eval(""))

print(results)
print(results.x_min)
print(results.f_min)

print(octave.eval('fvals'))
print(octave.eval('nfev'))
