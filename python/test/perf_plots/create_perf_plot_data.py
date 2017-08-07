from program import Program
from numpy import array_equal
from numpy import asarray
from utilities.sys_utils import createObject
from utilities import functions
from octave.pyfovec import dfovec
from octave.pydfox import dfoxs
# from threading import Timer

import pickle

from test.perf_plots.create_my_perf_data import runMyAlgorithm
from test.perf_plots.create_pyopt_perf_plot import runPyOpt

class Context:
	def __init__(self):
		self.nfev = 0
		self.fvals = []
	# 	self.timer = None
	#
	# def startDebugPrinter(self):
	# 	self.stopDebugPrinter()
	#
	# 	def print():
	# 		print('Currently at ' + str(self.nfev) + ' iterations.')
	# 	self.timer = Timer(5, print)
	#
	# def stopDebugPrinter(self):
	# 	if self.timer is None:
	# 		return
	# 	self.timer.cancel()


# Caches a repeated call at the same value
class FunctionEvaluationCacher:
	def __init__(self, orig):
		self.orig = orig
		self.cacheX = None
		self.cacheY = None

	def evaluate(self, x):
		if array_equal(self.cacheX, x) and False:
			return self.cacheY
		else:
			# Not thread safe...
			self.cacheX = x
			self.cacheY = self.orig(self.cacheX)
			return self.cacheY


def get_data(runner, n, m, nprob):
	context = Context()
	cacher = FunctionEvaluationCacher(lambda x: dfovec(m, n, x, nprob, context))
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
	inequalities.add(functions.Line(asarray([1, 1]), -100))
	inequalities.add(functions.Line(asarray([1, -1]), -100))
	inequalities.add(functions.Line(asarray([-1, 1]), -100))
	inequalities.add(functions.Line(asarray([-1, -1]), -100))

	dfovecProgram = Program('dfovec',
				objective,
				None,
				inequalities,
				dfoxs(n, nprob, 1))

	# context.startDebugPrinter()
	runner(dfovecProgram)
	# context.stopDebugPrinter()

	return context

def store_data(n, m, nprob):
	pyOptData = get_data(lambda program: runPyOpt(program), n, m, nprob)
	print(pyOptData.nfev)
	print(pyOptData.fvals)
	with open('runtimes/pyOpt_' + str(nprob) + ".p", "wb") as out:
		pickle.dump({'nfev': pyOptData.nfev, 'fvals': pyOptData.fvals}, out)

	myData = get_data(lambda program: runMyAlgorithm(program), n, m, nprob)
	print(myData.nfev)
	print(myData.fvals)
	with open('runtimes/mine_' + str(nprob) + ".p", "wb") as out:
		pickle.dump({'nfev': myData.nfev, 'fvals': myData.fvals}, out)

for nprob in range(1, 21):
	try:
		store_data(2, 2, nprob)
	except:
		pass


