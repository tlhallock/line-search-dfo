import json
import datetime
import os

import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *
from trust_region.dfo.trust_region.ellipse import Ellipse
import traceback


EXTREMELY_VERBOSE = False
ERRORS_COUNT = 0
ERRORS_FILE = 'images/errors/ellipse_error_{}_{}_params.json'
LOG_ERRORS = True


def log_error(params):
	global LOG_ERRORS
	global ERRORS_COUNT
	global ERRORS_FILE
	if not LOG_ERRORS:
		return
	ERRORS_COUNT += 1
	if not os.path.exists('images/errors'):
		os.makedirs('images/errors')
	with open(ERRORS_FILE.format(datetime.datetime.now(), ERRORS_COUNT), 'w') as output:
		json.dump(params.to_json(), output, indent=2)


def parse_params(js):
	params = EllipseParams()
	params.center = numpy.array(js['center'])
	params.A = numpy.array(js['A'])
	params.b = numpy.array(js['b'])
	params.include_point = numpy.array(js['include-point'])
	params.tolerance = js['tolerance']
	params.hot_start = numpy.array(js['hot-start']) if js['hot-start'] is not None else None
	return params


class EllipseParams:
	def __init__(self):
		self.center = None
		self.A = None
		self.b = None
		self.include_point = None
		self.tolerance = None
		self.hot_start = None

	def to_json(self):
		return {
			'center': [x for x in self.center],
			'A': [
				[self.A[r, c] for c in range(self.A.shape[1])]
				for r in range(self.A.shape[0])
			],
			'b': [b for b in self.b],
			'include-point': [i for i in self.include_point],
			'tolerance': self.tolerance,
			'hot-start': [s for s in self.hot_start] if self.hot_start is not None else None,
		}


def compute_maximal_ellipse(p):
	bbar = p.b - numpy.dot(p.A, p.center)

	# scale
	k = 1.0 / min(abs(bbar))
	if k < 1e-10 or k > 1e8:
		print("========================================================")
		print("in maximize ellipse, scaling factor would be:")
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

	si = None
	if p.include_point is not None:
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
	try:
		result = opt.solve(model)
	except Exception as e:
		traceback.print_exc()
		log_error(p)
		return False, None

	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
		log_error(p)
		return False, None
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")
		log_error(p)
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

	####################################################################################################################
	if EXTREMELY_VERBOSE:
		import random
		output_name = ''.join([random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(20)])
		from trust_region.util.history import Bounds
		from trust_region.util.plots import create_plot

		bounds = Bounds()
		bounds.extend(numpy.array([5, 3]))
		bounds.extend(numpy.array([-2, -2]))

		plot = create_plot('testing_ellipse', 'images/{}_debug_ellipse.png'.format(output_name), bounds)
		plot.ax.text(
			0.1, 0.1,
			str(ellipse.volume),
			horizontalalignment='center',
			verticalalignment='center',
			transform=plot.ax.transAxes
		)

		plot.add_polyhedron(p.A, p.b, label='bounds')
		plot.add_point(p.center, label='center', marker='x', color='r')
		if p.include_point is not None:
			plot.add_point(p.include_point, label='include', marker='+', color='y')
		ellipse.add_to_plot(plot)
		plot.save()

		with open('images/{}_debug_output.txt'.format(output_name), 'w') as output:
			output.write('A:' + '\n')
			output.write(str(p.A) + '\n')
			output.write('b:' + '\n')
			output.write(str(p.b) + '\n')
			output.write('pd 1:' + '\n')
			output.write(str(model.q[0]()) + '\n')
			output.write('pd 2:' + '\n')
			output.write(str(model.q[2]()) + '\n')
			output.write('pd 3:' + '\n')
			output.write(str(model.q[0]() * model.q[2]() - model.q[1]() * model.q[1]()) + '\n')
			output.write('include:' + '\n')
			if si is not None:
				output.write(str(
					-si[0] * model.q[1]() * si[1] + si[0] * model.q[2]() * si[0] +
					si[1] * model.q[0]() * si[1] - si[1] * model.q[1]() * si[0] -
					2 * model.q[0]() * model.q[2]() * k2 + 2 * model.q[1]() * model.q[1]() * k2
				) + '\n')
			output.write('hitting the boundaries' + '\n')
			for i in range(p.A.shape[0]):
				output.write(str(i) + '\n')
				output.write(str(
					p.A[i, 0] ** 2 * model.q[0]() +
					2 * p.A[i, 0] * p.A[i, 1] * model.q[1]() +
					p.A[i, 1] ** 2 * model.q[2]()
				) + '\n')
				output.write(str(bbar[i] * bbar[i] / 2) + '\n')

			output.write("centers" + '\n')
			output.write(str(ellipse.evaluate(p.center)) + '\n')
			if p.include_point is not None:
				output.write(str(ellipse.evaluate(p.include_point)) + '\n')
			output.write('done' + '\n')
	####################################################################################################################

	return True, ellipse


def compute_maximal_ellipse_after_shift(params, l1):
	params2 = EllipseParams()
	params2.center = (params.center - l1.center) / l1.radius
	params2.A = params.A * l1.radius
	params2.b = params.b - numpy.dot(params.A, l1.center)
	params2.include_point = (params.center - l1.center) / l1.radius if params.include_point is not None else None
	params2.tolerance = params.tolerance

	for i in range(params2.A.shape[0]):
		row_scale = 1.0 / abs(params2.b[i])
		if row_scale < 1e-12:
			row_scale = 1.0 / numpy.linalg.norm(A[i])
		params2.A[i] *= row_scale
		params2.b[i] *= row_scale

	# success1, ellipse1 = compute_maximal_ellipse(params)
	success2, ellipse2 = compute_maximal_ellipse(params2)
	if not success2:
		return False, None

	ellipse2.center = ellipse2.center * l1.radius + l1.center
	ellipse2.q = ellipse2.q / l1.radius / l1.radius
	ellipse2.q_inverse = ellipse2.q_inverse * l1.radius * l1.radius
	ellipse2.l = ellipse2.l / l1.radius
	ellipse2.l_inverse = ellipse2.l_inverse * l1.radius
	old_ds = ellipse2.ds
	ellipse2.ds = [d * l1.radius for d in old_ds]
	ellipse2.volume = numpy.pi / numpy.sqrt(numpy.linalg.det(ellipse2.q))
	return success2, ellipse2


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
