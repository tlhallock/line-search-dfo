
import numpy


import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *
from trust_region.util.ellipse import Ellipse


class EllipseParams:
	def __init__(self):
		self.center = None
		self.A = None
		self.b = None
		self.include_point = None
		self.center = None
		self.tolerance = None
		self.hot_start = None


def compute_maximal_ellipse(p):
	bbar = p.b - numpy.dot(p.A, p.center)

	# scale
	k = 1.0 / min(abs(bbar))
	if k < 1e-10 or k > 1e8:
		print("========================================================")
		print(k)
		print("========================================================")
		return False, None

	bbar = k * bbar

	model = ConcreteModel()
	model.q = Var(range(3))

	if p.hot_start is not None:
		for i in range(len(p.hot_start)):
			model.q[i].set_value(p.hot_start[i])

	model.constraints = ConstraintList()

	for i in range(p.A.shape[0]):
		# [A[i, 1], A[i, 2]] * [[q[0], q[1]], [q[1], q[2]]] * [A[i, 1], A[i, 2]].T
		# [A[i, 1], A[i, 2]] * [q[0] * A[i, 1] + q[1] * A[i, 2], q[1] * A[i, 1] + q[2] * A[i, 2]].T
		# A[i, 1] * (q[0] * A[i, 1] + q[1] * A[i, 2]) + A[i, 2] * (q[1] * A[i, 1] + q[2] * A[i, 2])
		# A[i, 1] * q[0] * A[i, 1] + A[i, 1] * q[1] * A[i, 2] + A[i, 2] *q[1] * A[i, 1] + A[i, 2] * q[2] * A[i, 2]
		# A[i, 1] ** 2 * q[0] + 2 * A[i, 1] * q[1] * A[i, 2] + A[i, 2] ** 2 * q[2]
		model.constraints.add(
			p.A[i, 0] ** 2 * model.q[0] +
			2 * p.A[i, 0] * p.A[i, 1] * model.q[1] +
			p.A[i, 1] ** 2 * model.q[2] <= bbar[i] * bbar[i] / 2
		)

	if p.include_point is not None and False:
		si = p.include_point - p.center
		# We want:
		#  0.5 * include.T Q include <= 1
		#  include.T ( q[0] q[1] ; q[1] q[2] )^-1 include <= 2
		#  include.T ( q[2] -q[1] ; -q[1] q[0] ) include <= 2 * q[0] * q[2] - 2 * q[1] ** 2
		#  0 <= 2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] -q[1] ; -q[1] q[0] ) include
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] -q[1] ; -q[1] q[0] ) include >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] * include[0] - q[1] * include[1] ; -q[1] * include[0] + q[0] * include[1]) >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - (include[0] * q[2] * include[0] - include[0] * q[1] * include[1]
		# 			 - include[1] * q[1] * include[0] + include[1] * q[0] * include[1]) >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include[0] * q[2] * include[0] + include[0] * q[1] * include[1]
		# 			 + include[1] * q[1] * include[0] - include[1] * q[0] * include[1] >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include[0] * q[2] * include[0] + 2 * include[0] * q[1] * include[1] - include[1] * q[0] * include[1] >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - q[2] * include[0] ** 2 + 2 * include[0] * q[1] * include[1] - q[0] * include[1] ** 2 >= 0

		#model.constraints.add(
		#	k ** 2 * 2 * model.q[0] * model.q[2] - k ** 2 * 2 * model.q[1] * model.q[1] -
		#	model.q[2] * si[0] ** 2 + 2 * si[0] * model.q[1] * si[1] - model.q[0] * si[1] ** 2 >= 0
		#)

		#model.constraints.add(
		#	si[0] * model.q[2] * si[0] - model.q[1] * si[1] - si[1] * model.q[1] * si[0] + model.q[0] * si[1]
		#)

		# s₀⋅⎝- q₁⋅s₁ + q₂⋅s₀⎠ + s₁⋅⎝q₀⋅s₁ - q₁⋅s₀⎠

		k2 = k * k
		model.constraints.add(
			-si[0] * model.q[1] * si[1] + si[0] * model.q[2] * si[0] +
			si[1] * model.q[0] * si[1] - si[1] * model.q[1] * si[0] -
			2 * model.q[0] * model.q[2] * k2 + 2 * model.q[1] * model.q[1] * k2 <= 0
		)

		# include[0] * q[2] * include[0] - include[0] * q[1] * include[1] - include[1] * q[1] * include[0] + include[1] * q[0] * include[1]
		# si[0] * model.q[2] * si[0] + -model.q[1] * si[1] + si[1] * -model.q[1] * si[0] + model.q[0] * si[1]                    +
		# 2 * q(1,1) * q(2,2) - 2 * q(1,2) * q(1,2) - q(2,2) * include(1) ** 2 + 2 * include(1) * model.q[1] * si[1] - model.q[0] * si[1] ** 2 >= 0

	# positive definite
	model.constraints.add(model.q[0] >= p.tolerance)
	model.constraints.add(model.q[2] >= p.tolerance)
	model.constraints.add(model.q[0] * model.q[2] - model.q[1] * model.q[1] >= p.tolerance)

	def objective_rule(m):
		return m.q[0] * m.q[2] - m.q[1] * m.q[1]
	model.objective = Objective(rule=objective_rule, sense=maximize)

	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
		return False, None
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")
		return False, None

	ellipse = Ellipse()
	ellipse.center = p.center
	ellipse.volume = -model.objective()
	ellipse.hot_start = [model.q[0](), model.q[1](), model.q[2]()]

	ellipse.q_inverse = numpy.array([
		[model.q[0](), model.q[1]()],
		[model.q[1](), model.q[2]()]
	])
	ellipse.q = numpy.linalg.inv(ellipse.q_inverse)

	ellipse.l = numpy.linalg.cholesky(ellipse.q).T
	ellipse.l_inverse = numpy.linalg.inv(ellipse.l)

	ellipse.q_inverse = ellipse.q_inverse / (k * k)
	ellipse.q = k * k * ellipse.q

	ellipse.l = k * ellipse.l
	ellipse.l_inverse = ellipse.l_inverse / k

	ellipse.ds = []
	ellipse.lambdas = []
	for i in range(p.A.shape[0]):
		ar = p.A[i, :]
		br = bbar[i] / k
		qa = numpy.dot(ellipse.q_inverse, ar)
		lmbda = br / numpy.dot(ar, qa)
		d = lmbda * qa
		ellipse.lambdas.append(lmbda)
		ellipse.ds.append(d)

	return True, ellipse

# q0 - l q1
# q1     q2 - l
# (q0 - l) * (q2 - l) - q1 * q1 == 0
# (q0 - l) * q2 - (q0 - l) * l - q1 * q1 == 0
# q0 * q2 - l* q2 - (q0 * l - l* l) - q1 * q1 == 0
# q0 * q2 - l * q2 - q0 * l + l * l - q1 * q1 == 0
# l * l - (q0 + q1)l  + q0 * q2  - q1 * q1 == 0

# want:
# b ** 2 - 4ac >= 0
# (-b - sqrt(b ** 2 - 4ac)) / 2a >= 0

# b ** 2 - 4c >= 0
# -b - sqrt(b ** 2 - 4c) >= 0

# b ** 2 >= 4c
# - sqrt(b ** 2 - 4c) >= b

# b ** 2 >= 4c
# b ** 2 - 4c >= b ** 2

# b ** 2 >= 4c
# b ** 2 - 4c >= b ** 2
