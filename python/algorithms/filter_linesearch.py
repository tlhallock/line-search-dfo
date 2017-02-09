from math import inf as infinity
from numpy import int as integral

from numpy        import bmat      as blockmat
from numpy        import concatenate
from numpy        import dot
from numpy        import empty
from numpy        import zeros
from numpy.linalg import cond      as condition_number
from numpy.linalg import lstsq
from numpy.linalg import norm      as norm
from numpy.linalg import solve     as linsolve
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from utilities.nondom import NonDomSet


class Constants:
	def __init__(self, theta_max):
		self.theta_max   = theta_max					# (theta(x0), infty)
		self.gamma_theta = .01  						# (0,1)
		self.gamma_f     = .75  						# (0,1)
		self.delta       = .01  						# (0,infty)
		self.gamma_alpha = .5   						# (0,1]
		self.s_theta     = 2    						# (1,infty)
		self.s_f         = 3    						# [1,infty)
		self.eta_f       = .025  						# (0, .5)
		self.tau_one     = .25  						# (0, tau_two]
		self.tau_two     = .75  						# [tau_two, 1)
		self.plot        = True 						# Should all the plots be generated
		self.max_condition_number = 1000

		self.tau = (self.tau_one + self.tau_two) / 2


class Result:
	def __init__(self):
		self.number_of_iterations = -1 # we increment by one in the first iteration
		self.restorations = 0
		self.ftype_iterations = 0
		self.filter_modified_count = 0
		self.pareto = NonDomSet()
		self.success = False
		self.f_min = infinity
		self.x_min = 0
		self.filterRejectedCount = 0
		self.criteria_satifisfied_but_trust_region_not = 0

	def newF(self, otherX, otherF):
		if self.f_min < otherF:
			return
		self.f_min = otherF
		self.x_min = otherX



def theta(statement, x):
	return getThetaAndIneq(statement, x)[0]

def getThetaAndIneq(statement, x):
	c, _, ineq = getConstraintInfo(statement, x)
	return norm(c), ineq
	# retVal = 0
	# if statement.hasEqualityConstraints():
	# 	retVal += norm(statement.equalityConstraints(x))
	# if statement.hasInequalityConstraints():
	# 	c = statement.inequalityConstraints(x)
	# 	retVal += norm(c[c > -statement.tol])
	# return retVal

def getConstraintInfo(statement, x):
	if statement.hasEqualityConstraints():
		cEq = statement.equalityConstraints(x)
		aEq = statement.equalityConstraintsJacobian(x)

		if not statement.hasInequalityConstraints():
			return cEq, aEq, empty(0)

	if statement.hasInequalityConstraints():
		cIneqAll = statement.inequalityConstraints(x)
		aIneqAll = statement.inequalityConstraintsJacobian(x)

		active = cIneqAll > -statement.tol
		cIneqActive = cIneqAll[active]
		aIneqActive = aIneqAll[active]

		if statement.hasEqualityConstraints():
			c = concatenate([cEq, cIneqActive])
			A = blockmat([[aEq], [aIneqActive]])
			return c, A, cIneqAll
		else:
			return cIneqActive, aIneqActive, cIneqAll

	return None, None, None

# don't check constraints that are currently active going to false...
def addedActiveConstraint(newIneq, cIneq, tol):
	# Check that we are not adding any active constraints...
	# Don't want to just check the "active" variable from computeConstraintInfo
	# because of the tolerance issue while we are on it.
	# addedInactive = all([not newIneq[i] for i, x in enumerate(state.cIneq) if not x])
	# comparing with zero instead of tolerance (not entirely sure why...)
	# I might should use -tol...
	return any([newIneq[i] > 0 for i, x in enumerate(cIneq) if x < tol])

class AlgorithmState:
	def __init__(self, statement):
		self.x      = statement.x0
		self.grad   = 0
		self.pareto = NonDomSet()
		self.f      = infinity
		self.grad   = None
		self.hess   = None
		self.A      = None
		self.c      = None
		self.cIneq  = None
		self.d      = empty(len(self.x))
		self.x_new  = None
		self.ftype  = False
		self.accept = False
		self.theta  = None

	def setCurrentIterate(self, statement):
		self.f		= statement.objective(self.x)
		self.grad	= statement.gradient(self.x)
		self.hess	= statement.hessian(self.x)
		self.theta  = theta(statement, self.x)

		self.c, self.A, self.cIneq = getConstraintInfo(statement, self.x)

	def createKKT(self):
		if self.A is None:
			return self.hess
		m = self.getM()
		return blockmat([[self.hess, self.A.T], [self.A, zeros((m, m))]])

	def createRhs(self):
		if self.A is None:
			return self.grad

		return concatenate([self.grad, self.c])

	def getN(self):
		return len(self.x)

	def getM(self):
		if self.A is None:
			return 0
		return self.A.shape[0]

	def show(self, statement):
		fileName = statement.createBasePlotAt(self.x)

		#amin(shifted, 0)
		totalDist = norm(self.x_new - self.x)
		hw = .1 * totalDist
		hl = .1 * totalDist

		plt.arrow(x=self.x[0], y=self.x[1],
				dx = (self.x_new[0] - self.x[0]), dy = (self.x_new[1] - self.x[1]),
				head_width = hw, head_length = hl, fc = 'g', ec = 'g')

		plt.arrow(x=self.x[0], y=self.x[1],
				dx = -totalDist * self.grad[0] / norm(self.grad),
				dy = -totalDist * self.grad[1] / norm(self.grad),
				head_width = hw, head_length = hl, fc = 'y', ec = 'y')

		plt.savefig(fileName)
		plt.close()


def checkStoppingCriteria(statement, state):
	hasConstraints = statement.hasEqualityConstraints() or statement.hasInequalityConstraints()

	if not hasConstraints:
		return norm(state.grad) < statement.tol

	if statement.hasConstraints() and (state.c > statement.tol).any():
		return False


	lmbda,_,_,_ = lstsq(state.A.T, -state.grad)
	# What exactly is this supposed to be checking?
	if norm(state.grad + dot(state.A.T, lmbda)) > statement.tol:
		return False

	if statement.hasInequalityConstraints():
		numEqualityConstraints = statement.getNumEqualityConstraints()
		if any(lmbda[numEqualityConstraints:len(lmbda)] < -statement.tol):
			return False

	return True




def compute_alpha_min(statement, constants, state):
	gDotd = dot(state.grad.T, state.d)
	if gDotd < -statement.tol:
		return constants.gamma_alpha * min(
			constants.gamma_theta,
			-constants.gamma_f*state.theta/(gDotd),
			(constants.delta*state.theta**constants.s_theta)/((-gDotd)**constants.s_f))
	else:
		return constants.gamma_alpha * constants.gamma_theta





def restore_feasibility(statement, x0):
	res = minimize(lambda x: theta(statement, x)[0], x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': False, 'maxfev': 1000})
	return res.x


def filter_line_search(program, constants):
	results = Result()
	state = AlgorithmState(program)
	
	while True:
		results.number_of_iterations += 1
		print(results.number_of_iterations)

		state.setCurrentIterate(program)

		n = state.getN()

		if checkStoppingCriteria(program, state):
			if not program.converged():
				results.criteria_satifisfied_but_trust_region_not += 1
				continue
			results.newF(state.x, state.f)
			results.success = True
			break

		kktmat = state.createKKT()
		state.cond = condition_number(kktmat)
		if state.cond > constants.max_condition_number:
			results.restorations += 1
			state.x = restore_feasibility(program, state.x)
			continue

		rhs = state.createRhs()
		vec = linsolve(kktmat, rhs.T)
		state.d[:] = -vec[0:n]

		state.alpha_min = compute_alpha_min(program, constants, state)
		state.alpha = 1
		state.accept = False

		gDotd = dot(state.grad.T, state.d)

		while not state.accept:
			m = state.alpha * gDotd

			# Hack, maybe: clip to trust region: this should be solved in the subproblem!!!
			state.d = program.clipToTrustRegion(state.d)

			if state.alpha < state.alpha_min:
				state.x = restore_feasibility(program, state.x)
				results.restorations += 1
				break

			state.x_new = state.x + state.alpha * state.d
			state.theta_new, newIneq = getThetaAndIneq(program, state.x_new)
			state.f_new = program.objective(state.x_new)

			# If we are about to add a constraint that was not active, then don't
			if addedActiveConstraint(newIneq, state.cIneq, program.tol):
				state.alpha = state.alpha * constants.tau
				continue


			if constants.plot:
				state.show(program)

			if results.pareto.is_dominated((state.theta_new, state.f_new)):
				state.alpha = state.alpha * constants.tau
				results.filterRejectedCount += 1
				continue

			state.ftype = m < 0 and ((-m)**constants.s_f * state.alpha**(1-constants.s_f) > constants.delta * state.theta ** constants.s_theta);
			if state.ftype:
				if state.f_new <= state.f + constants.eta_f * m:
					state.accept = True
					break
			else:
				eight_a = state.theta_new <= (1-constants.gamma_theta) * state.theta
				eight_b = state.f_new <= state.f - constants.gamma_f * state.theta_new
				if eight_a or eight_b:
					state.accept = True
					break
			state.alpha = state.alpha * constants.tau

		if state.accept:
			if not program.acceptable(state.x_new):
				continue

			if state.ftype:
				results.ftype_iterations += 1
				if (1-constants.gamma_theta) * state.theta_new > program.tol:
					results.pareto.add(((1 - constants.gamma_theta) * state.theta_new, state.f - constants.gamma_f * state.theta_new))
					results.filter_modified_count += 1
			state.x = state.x_new

	return results
