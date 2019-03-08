
import test_nlp
import numpy as np
from .polyhedron import Polyhedron


'''

f2py PROB.FOR CONV.FOR CALL.FOR -c -m test_nlp only: get_description evaluate_objective evaluate_constraints evaluate_constraints_gradient evaluate_objective_gradient

cp test_nlp.cpython-36m-x86_64-linux-gnu.so /work/research/line-search-dfo/pyomo/

'''


class HsProblem:
	def __init__(self, problem_no):
		self.problem_no = problem_no
		test_nlp.l8.ntp = self.problem_no
		test_nlp.get_description()
		# dimension
		self.n = int(test_nlp.l1.n)
		# number of linear inequality constraints
		self.nili = int(test_nlp.l1.nili)
		# number of nonlinear inequality constraints
		self.ninl = int(test_nlp.l1.ninl)
		# number of linear equality constraints
		self.neli = int(test_nlp.l1.neli)
		# number of nonlinear equality constraints
		self.nenl = int(test_nlp.l1.nenl)
		self.x0 = np.array(test_nlp.l2.x)[:self.n].copy()
		self.lb = np.array([test_nlp.l13.xl[i] if test_nlp.l11.lxl[i] else -np.infty for i in range(self.n)])
		self.ub = np.array([test_nlp.l14.xu[i] if test_nlp.l12.lxu[i] else +np.infty for i in range(self.n)])
		self.number_of_local_optimum = test_nlp.l20.nex if test_nlp.l20.lex else None
		self.x_star = np.array(test_nlp.l20.xex)[:self.n].copy()
		self.f_star = float(test_nlp.l20.fex)

	@property
	def m(self):
		return self.nili + self.ninl + self.neli + self.nenl

	def evaluate_constraints(self, x):
		test_nlp.l2.x[:self.n] = x[:]
		test_nlp.l9.index1[:] = True
		test_nlp.evaluate_constraints()
		return np.array(test_nlp.l3.g)[:self.m].copy()

	def evaluate_objective(self, x):
		test_nlp.l2.x[:self.n] = x[:]
		test_nlp.evaluate_objective()
		return float(test_nlp.l6.fx)

	def evaluate_constraints_gradient(self, x):
		test_nlp.l10.index2[:] = True
		test_nlp.l2.x[:self.n] = x[:]
		test_nlp.evaluate_constraints_gradient()
		return np.array(test_nlp.l5.gg)[:self.m, :self.n].copy()

	def determine_constraints(self):
		A = -self.evaluate_constraints_gradient(np.zeros(self.n))
		b = self.evaluate_constraints(np.zeros(self.n))

		x_trial = 10 * np.random.random(self.n)
		error = np.linalg.norm(b - np.dot(A, x_trial) - self.evaluate_constraints(x_trial))
		print('linearity test error', error)
		print('found constraints matrix', A)
		if error > 1e-10:
			raise Exception('Constraints are not linear')
		return Polyhedron(A, b)

	def get_initial_feasibility(self, tol=1e-10):
		has_active = False
		for gi in self.evaluate_constraints(self.x0):
			if gi > tol:
				return 'infeasible'
			if gi < -tol:
				has_active = True
		for i in range(self.n):
			if self.x0[i] < self.lb[i] - tol:
				return 'infeasible'
			if self.x0[i] < self.lb[i] + tol:
				has_active = True
			if self.x0[i] > self.ub[i] + tol:
				return 'infeasible'
			if self.x0[i] > self.ub[i] - tol:
				has_active = True

		return 'active' if has_active else 'feasible'

	def to_json(self):
		return {
			"problem_number": self.problem_no,
			"m": self.m,
			"n": self.n,
			"number-of-linear-inequality-constraints": self.nili,
			"number-of-nonlinear-inequality-constraints": self.ninl,
			"number-of-linear-equality-constraints": self.neli,
			"number-of-nonlinear-equality-constraints": self.nenl,
			"x0": self.x0,
			"lb": self.lb,
			"ub":  self.ub,
			"minimizer": self.x_star,
			"minimum": self.f_star
		}
