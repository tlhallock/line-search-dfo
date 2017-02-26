from math import inf as infinity
from numpy import int as integral
from dfo import dfo_model

from dfo import polynomial_basis

from numpy import reshape
from numpy import bmat      as blockmat
from numpy import asarray
from numpy import concatenate
from numpy import dot
from numpy import empty
from numpy import zeros
from numpy import arange
from numpy.linalg import cond      as condition_number
from numpy.linalg import lstsq
from numpy.linalg import norm      as norm
from numpy.linalg import solve     as linsolve
from numpy.linalg import pinv
from scipy.optimize import minimize
from scipy.optimize import linprog
import matplotlib.pyplot as plt


from numpy import setdiff1d


from utilities.nondom import NonDomSet


class Constants:
	def __init__(self):
		self.delta = 1
		self.gamma_0 = .1
		self.gamma_1 = .5
		self.gamma_2 = 2
		self.eta_1 = .9
		self.eta_2 = .9
		self.gamma_theta = 1e-4
		self.kappa_delta = .7
		self.kappa_theta = 1e-4
		self.kappa_mu = 100
		self.mu = .01
		self.psi = 2
		self.kappa_tmd = .01


class Results:
	def __init__(self):
		self.number_of_iterations = 0
		self.restorations = 0
		self.filter_modified_count = 0
		self.success = False
		self.f_min = infinity
		self.x_min = 0
		self.filterRejectedCount = 0

	def newF(self, otherX, otherF):
		if self.f_min < otherF:
			return
		self.f_min = otherF
		self.x_min = otherX


def theta(cEq, cIneq, active=None):
	if active is None:
		raise Exception('Not implemented.')
	s = 0
	s += norm(cEq)
	s += norm(cIneq[active])
	return s



def _createModelFunction(program, radius, xsi):
	b = polynomial_basis.PolynomialBasis(len(program.x0), 2)

	equalityIndices = empty(program.getNumEqualityConstraints(), dtype=int)
	inequalityIndices = empty(program.getNumInequalityConstraints(), dtype=int)

	funs = []
	index = 0

	funs.append(program.f)
	objectiveIndex = int(index)
	index += 1

	for i in range(len(equalityIndices)):
		funs.append(program.eq.getFunction(i))
		equalityIndices[i] = int(index)
		index += 1
	for i in range(len(inequalityIndices)):
		funs.append(program.ineq.getFunction(i))
		inequalityIndices[i] = int(index)
		index += 1

	model = dfo_model.MultiFunctionModel(funs, b, program.x0, radius, xsi)

	return (model, objectiveIndex, equalityIndices, inequalityIndices)

# OMG, I can't figure out how I am calling this incorrectly, but with an infeasible problem the method is returning a result
def dbl_check_sol(cons, res):
	if not res.success:
		return True
	for c in cons:
		if c['type'] == 'ineq':
			if (c['fun'](res.x) < -1e-8).any():
				return False
		elif c['type'] == 'eq':
			if norm(c['fun'](res.x)) > 1e-8:
				return False
		else:
			raise Exception('unknown type of constraint')
	return True


class AlgorithmState:
	def __init__(self, statement, constants):
		self.tol = statement.tol
		self.x = statement.x0

		# the model function:
		self.model, _, self.equalityIndices, self.inequalityIndices = _createModelFunction(statement, constants.delta, 1e-3)

		# model function
		self.mf = None
		self.mh = None
		self.mg = None

		# the filter
		self.pareto = NonDomSet()

		# Information of objective
		self.f = infinity
		self.grad = None
		self.hess = None

		# Information of constraints
		self.theta = None
		self.A = None
		self.c = None

		self.cIneq = None
		self.AIneq = None
		self.cEq = None
		self.AEq = None
		self.active = None

		# The hessian of the lagrangian
		self.H = None

		# The current step
		self.t = empty(len(self.x))
		self.n = empty(len(self.x))

		# The current step
		self.n = None
		self.t = None
		self.s = None

	def computeCurrentValues(self, program):
		self.n = None
		self.t = None
		self.s = None

		self.model.computeValueFromDelegate(self.x)
		self.model.setNewModelCenter(self.x)
		self.model.improve(program.getNextPlotFile('improve'))

		self.mf = self.model.getQuadraticModel(0)
		self.mh = self.model.getQuadraticModels(self.equalityIndices)
		self.mg = self.model.getQuadraticModels(self.inequalityIndices)

		self.f = self.mf.evaluate(self.x)
		self.grad = self.mf.gradient(self.x)
		self.hess = self.mf.hessian(self.x)

		self.cEq = self.mh.evaluate(self.x)
		self.AEq = self.mh.jacobian(self.x)

		self.cIneq = self.mg.evaluate(self.x)
		self.AIneq = self.mg.jacobian(self.x)

		self.computeActiveConstraints()
		self.computeHessianOfLagrangian()

		self.theta = theta(self.cEq, self.cIneq, self.active)

	def computeHessianOfLagrangian(self):
		n = len(self.x)
		m = self.AEq.shape[0] + self.AIneq.shape[0]
		FULL_KKT = blockmat([
			[self.hess, self.AEq.T, self.AIneq.T],
			[self.AEq,   zeros([self.AEq.shape[0], m])],
			[self.AIneq, zeros([self.AIneq.shape[0], m])]])
		FULL_RHS = concatenate([-self.grad, self.cEq, self.cIneq])
		if condition_number(FULL_KKT) > 1000:
			# results.restorations += 1
			# state.x = restore_feasibility(program, state.x)
			print('Inverting singular matrix to find lagrange multipliers.')

		#vec = linsolve(FULL_KKT, FULL_RHS.T)
		vec = lstsq(FULL_KKT, FULL_RHS.T)[0]
		# newton_direction   = -vec[:n]
		lagrange_polynomials = -vec[n:]

		qs = self.model.getQuadraticModels(arange(1,m+1)).hessian(self.x)
		self.H = self.hess

		# I am sure I could use einstien summation notation or something
		for i in range(len(lagrange_polynomials)):
			self.H += lagrange_polynomials[i] * qs[i]


	def computeActiveConstraints(self):
		if not len(self.cEq) == 0 and len(self.cIneq) == 0:
			self.active = empty(0)
			self.A = self.AEq
			self.c = self.AEq
			return

		if not len(self.cIneq) == 0:
			self.active = self.cIneq > -self.tol
			cIneqActive = self.cIneq[self.active]
			aIneqActive = self.AIneq[self.active]

			if len(self.cEq) == 0:
				self.c = cIneqActive
				self.A = aIneqActive
			else:
				self.c = concatenate([self.cEq, cIneqActive])
				self.A = blockmat([[self.AEq], [aIneqActive]])
			return

		self.c = empty(0)
		self.A = empty([0, 0])
		self.active = empty(0)


	#
	# def _createKKT(self):
	# 	if self.A is None:
	# 		return self.hess
	# 	m = self.getM()
	# 	return blockmat([[self.hess, self.A.T], [self.A, zeros((m, m))]])
	#
	# def _createRhs(self):
	# 	if self.A is None:
	# 		return self.grad
	# 	return concatenate([self.grad, self.c])

	def computeChi(self):
		if self.n is None:
			return None, False
		result = linprog(c=self.grad + dot(self.H, self.n),
						 A_ub= self.AIneq, b_ub= -self.cIneq,
						 A_eq=self.AEq, b_eq=zeros(self.AEq.shape[0]))
		if result.success:
			return abs(result.fun), True
		else:
			return None, False

	def computeNormalComponent(self):
		if True:
			tr_jac_dim=(1, self.getN())
			initialN = -asarray(dot(dot(self.A.T, pinv(dot(self.A, self.A.T))), self.c)).flatten() - self.x
			cons = [{'type': 'ineq',
					 'fun': lambda n: -self.cIneq - dot(self.AIneq, n),
					 'jac': lambda n: -self.AIneq},
					{'type': 'ineq',
					 'fun': lambda n: self.model.modelRadius**2 - dot(n, n),
					 'jac': lambda n: reshape(-2*n, tr_jac_dim)},
					{'type': 'eq',
					 'fun': lambda n: self.cEq + dot(self.AEq, n),
					 'jac': lambda n: self.AEq}]
			res_cons = minimize(lambda n: dot(n, n), jac=lambda n: 2 * n, x0=initialN,
								constraints=cons, method='SLSQP', options={"disp": False, "maxiter": 1000}, tol=self.tol)
			if res_cons.success and dbl_check_sol(cons, res_cons):
				self.n = res_cons.x
				pass
			else:
				self.n = None
		elif False:
			self.n = -asarray(dot(dot(self.A.T, pinv(dot(self.A, self.A.T))), self.c)).flatten() - self.x
		else:
			self.n = -asarray(dot(dot(self.A.T, pinv(dot(self.A, self.A.T))), self.c)).flatten() - self.x

			# Performing line search to satisfy others.
			# Not pretty
			inactive = self.cIneq <= -self.tol
			if not inactive.any():
				return

			inactiveC = self.cIneq[inactive]
			inactiveA = self.AIneq[inactive]

			if inactiveC + dot(inactiveA, self.x + self.n) < self.tol:
				return

			alpha = 1
			max_step = 1
			min_step = 0
			while max_step - min_step > self.tol:
				alpha = (max_step + min_step) / 2
				if max(inactiveC + dot(inactiveA, self.x + alpha * self.n)) > self.tol:
					max_step = alpha
				else:
					min_step = alpha

			self.n = min_step * self.n


	def computeTangentialStep(self):
		# for a quadratic program:
		rhs1 = self.cIneq + dot(self.AIneq, self.n)

		cons = [{'type': 'ineq',
					'fun': lambda t: -rhs1 - dot(self.AIneq, t),
					'jac': lambda t: -self.AIneq},
				# Adding a trust region boundary to it
				# This is the wrong trust region boundary
				{'type': 'ineq',
					 'fun': lambda t: self.model.modelRadius**2 - dot(t, t),
					 'jac': lambda t: reshape(-2*t, (1, 2))},
				{'type': 'eq',
				 	'fun': lambda t: dot(self.AEq, t),
				 	'jac': lambda t: self.AEq}]
		res_cons = minimize(
			lambda t: dot(self.grad + dot(self.H, self.n), t) + .5 * dot(t.T, dot(self.H, t)),
			jac= lambda t: dot(self.H, self.n) + dot(self.H, t),
			x0=zeros(len(self.x)), constraints=cons, method='SLSQP', options={"disp": False, "maxiter": 1000}, tol=self.tol)

		if res_cons.success and dbl_check_sol(cons, res_cons):
			self.t = res_cons.x
			return res_cons.x, True
		else:
			self.t = None
			return None, False

	def getN(self):
		return len(self.x)

	def getM(self):
		if self.A is None:
			return 0
		return self.A.shape[0]

	def getPlotRadius(self):
		maxDist = max(norm(self.model.unshifted - self.x, axis=1))
		if self.n is not None:
			maxDist = max(maxDist, norm(self.n))
		if self.t is not None:
			maxDist = max(maxDist, norm(self.t))
		if self.s is not None:
			maxDist = max(maxDist, norm(self.s - self.x))
		maxDist = max(maxDist, self.model.modelRadius)
		return maxDist * 1.2

	def show(self, statement, action):
		center = self.x
		radius = self.getPlotRadius()
		statement.createBasePlotAt(center, radius)

		self.model.addPointsToPlot(center, radius)

		# amin(shifted, 0)
		totalDist = radius
		hw = .1 * totalDist
		hl = .1 * totalDist

		if self.s is not None:
			plt.arrow(x=self.x[0], y=self.x[1],
					  dx=(self.s[0]), dy=(self.s[1]),
					  head_width=hw, head_length=hl, fc='g', ec='g')

		if self.n is not None:
			plt.arrow(x=self.x[0], y=self.x[1],
					  dx=self.n[0], dy=self.n[1],
					  head_width=hw, head_length=hl, fc='r', ec='r')

		if self.t is not None:
			plt.arrow(x=self.x[0], y=self.x[1],
					  dx=self.t[0], dy=self.t[1],
					  head_width=hw, head_length=hl, fc='b', ec='b')

		plt.arrow(x=self.x[0], y=self.x[1],
				  dx=-totalDist * self.grad[0] / norm(self.grad),
				  dy=-totalDist * self.grad[1] / norm(self.grad),
				  head_width=hw, head_length=hl, fc='y', ec='y')

		plt.savefig(statement.getNextPlotFile(action))
		plt.close()

	def delta(self):
		return self.model.modelRadius

	def decreaseRadius(self, constants):
		self.model.multiplyRadius((constants.gamma_0 + constants.gamma_1) / 2)

	def increaseRadius(self, constants):
		self.model.multiplyRadius((1 + constants.gamma_2) / 2)


def restore_feasibility(state):
	def obj(x):
		ineq = state.mg.evaluate(x)
		eq   = state.mh.evaluate(x)
		active = ineq > -state.tol
		return theta(eq, ineq, active)

	res = minimize(obj, state.x, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': False, 'maxfev': 1000})

	if res.success:
		return res.x, True
	else:
		return None, False










def trust_filter(program, constants):
	results = Results()
	state = AlgorithmState(program, constants)

	while True:
		results.number_of_iterations += 1

		state.computeCurrentValues(program)

		state.computeNormalComponent()
		state.show(program, 'normal_step')

		# This is not the correct feasible region to check non-emptyness!
		chi, nonempty = state.computeChi()
		if nonempty:
			print("current theta  = " + str(state.theta))
			print("current chi    = " + str(chi))
			print("current radius = " + str(state.delta()))
			print("---------------------------------------")

		if state.theta < program.tol and chi < program.tol:
			if state.delta() < program.tol:
				results.newF(state.x, state.f)
				results.success = True
				return results
			state.decreaseRadius(constants)
			continue

		if state.n is not None and norm(state.n) < constants.kappa_delta * state.delta() * min(1, constants.kappa_mu * state.delta() ** constants.mu):
			state.computeTangentialStep()

			# This check was not in the paper...
			if state.t is None:
				results.restorations += 1
				newx, possible = restore_feasibility(state)
				if not possible:
					results.success = False
					break
				state.x = newx
				continue

			state.s = state.t + state.n
			state.x_new = state.x + state.s
			state.show(program, 'tangential_step')

			expectedY  = state.model.interpolate(state.x_new)
			actualY, _ = state.model.computeValueFromDelegate(state.x_new)

			violation_eq   = actualY[state.equalityIndices]
			violation_ineq = actualY[state.inequalityIndices]
			active = violation_ineq > -program.tol
			theta_new = theta(violation_eq, violation_ineq, active)

			fnew = actualY[0]
			fold = expectedY[0]
			if not state.pareto.is_dominated((theta_new, fnew)):
				if state.f - fold >= constants.kappa_theta * state.theta ** constants.psi or True:
					#rho = (state.f - fnew) / (state.f - state.model.interpolate(state.x_new))
					rho = (state.f - fnew) / (state.model.interpolate(state.x)[0] - state.model.interpolate(state.x_new)[0])
					if rho < constants.eta_1:
						state.decreaseRadius(constants)
						# ensure poisedness
						continue
					if rho > constants.eta_2:
						if norm(state.s) < state.delta():
							state.decreaseRadius(constants)
						else:
							state.increaseRadius(constants)
				else:
					state.pareto.add(
		 				((1 - constants.gamma_theta) * theta_new, state.f - constants.gamma_theta * theta_new))
			else:
				state.decreaseRadius(constants)
				continue
		else:
			results.restorations += 1
			newx, possible = restore_feasibility(state)
			if not possible:
				results.success = False
				break
			state.x = newx

	return results