
from pyOpt import Optimization
from pyOpt import SDPEN

def translateProgramToPyOpt(dfovecProgram):
	# Currently only handles inequality
	def objfunc(x):
		f = dfovecProgram.objective(x)

		g = []
		if dfovecProgram.hasInequalityConstraints():
			g = dfovecProgram.inequalityConstraints(x)

		fail = 0
		return f, g, fail

	opt_prob = Optimization('Dfovec problem', objfunc)
	for i in range(len(dfovecProgram.x0)):
		opt_prob.addVar('x' + str(i), lower=-1000.0, upper=1000.0, value=dfovecProgram.x0[i])
	opt_prob.addObj('f')

	numIneq = dfovecProgram.getNumInequalityConstraints()
	opt_prob.addConGroup('g', numIneq, type='i', lower=[-10000] * numIneq, upper=[0] * numIneq)

	print(opt_prob)
	return opt_prob


def runPyOpt(dfovecProgram):
	opt_prob = translateProgramToPyOpt(dfovecProgram)
	# Instantiate Optimizer (SDPEN) & Solve Problem
	sdpen = SDPEN()
	# sdpen.setOption('iprint', -1)
	sdpen(opt_prob)
	print(opt_prob.solution(0))

