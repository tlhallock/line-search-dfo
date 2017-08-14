from program import Program
from numpy import array_equal
from numpy import eye
from numpy import zeros
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
	initialX = dfoxs(n, nprob, 1)
	initialY = dfovec(m, n, initialX, nprob)
	shiftConstraintsBy = 2
	radius = 50

	context = Context()
	cacher = FunctionEvaluationCacher(lambda x: dfovec(m, n, x, nprob, context) - initialY - shiftConstraintsBy)
	# an object with an evaluate method
	objective = createObject()
	objective.evaluate = lambda x: cacher.evaluate(x)[0]
	objective.getOutDim = lambda: 1
	objective.getFunction = lambda idx: objective
	# an object with an evaluate method
	dFovecInequality = createObject()
	dFovecInequality.evaluate = lambda x: cacher.evaluate(x)[1:]
	dFovecInequality.getOutDim = lambda: m - 1
	dFovecInequality.getFunction = lambda idx: dFovecInequality

	inequalities = functions.VectorFunction(n)
	inequalities.add(functions.Quadratic(eye(n), zeros(n), -radius * radius))

	dfovecProgram = Program('dfovec_' + str(nprob),
				objective,
				None,
				inequalities,
				initialX,
				max_iters = 1000)

	runner(dfovecProgram)

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

problems = [{
		'nprob': 1,
		'm': 2,
		'n': 2
	}, {
		'nprob': 2,
		'm': 2,
		'n': 2
	}, {
		'nprob': 3,
		'm': 2,
		'n': 2
	}, {
		'nprob': 4,
		'm': 2,
		'n': 2
	}, {
		'nprob': 5,
		'm': 3,
		'n': 3
	}, {
		'nprob': 6,
		'm': 4,
		'n': 4
	}, {
		'nprob': 7,
		'm': 2,
		'n': 2
	}, {
		'nprob': 8,
		'm': 18,
		'n': 4
	}, {
		'nprob': 9,
		'm': 11,
		'n': 4
	}, {
		'nprob': 10,
		'm': 16,
		'n': 3
	}, {
		'nprob': 11,
		'm': 31,
		'n': 2
	}, {
		'nprob': 12,
		'm': 3,
		'n': 3
	# large chi even though there is a small radius
	# }, {
	# 	'nprob': 13,
	# 	'm': 2,
	# 	'n': 2
	}, {
		'nprob': 14,
		'm': 4,
		'n': 4
	}, {
		'nprob': 15,
		'm': 2,
		'n': 2
	}, {
		'nprob': 16,
		'm': 2,
		'n': 2
	}, {
		'nprob': 17,
		'm': 33,
		'n': 5
	# takes a while
	# }, {
	# 	'nprob': 18,
	# 	'm': 65,
	# 	'n': 11
	}, {
		'nprob': 19,
		'm': 2 + 4,
		'n': 2
	}, {
		'nprob': 20,
		'm': 2,
		'n': 2
	}, {
		'nprob': 21,
		'm': 8,
		'n': 8
	},
]

for i in range(len(problems)):
	# if problems[i]['nprob'] != 1:
	# 	continue
	store_data(problems[i]['n'], problems[i]['m'], problems[i]['nprob'])


# get_data(lambda program: runMyAlgorithm(program), 2, 2, 1)



