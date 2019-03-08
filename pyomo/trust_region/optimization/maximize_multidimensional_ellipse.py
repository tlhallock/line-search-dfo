import datetime
import os

import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *
from trust_region.dfo.trust_region.ellipse import Ellipse
from trust_region.util.list_matrices import vector_to_matrix
from trust_region.util.list_matrices import transpose
from trust_region.util.list_matrices import multiply
from trust_region.util.list_matrices import determinant
from trust_region.util.polyhedron import parse_polyhedron
from trust_region.util.polyhedron import Polyhedron

from trust_region.util.utils import write_json


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
		write_json(params.to_json(), output)


def parse_params(js):
	params = EllipseParams()
	params.center = numpy.array(js['center'])
	params.polyhedron = parse_polyhedron(js['polyhedron'])
	# params.include_point = numpy.array(js['include-point'])
	params.tolerance = js['tolerance']
	params.hot_start = numpy.array(js['hot-start']) if js['hot-start'] is not None else None
	return params


class EllipseParams:
	def __init__(self):
		self.center = None
		self.polyhedron = None
		self.tolerance = None
		self.hot_start = None

	def to_json(self):
		return {
			'center': self.center,
			'polyhedron': self.polyhedron.to_json(),
			'tolerance': self.tolerance,
			'hot-start': self.hot_start,
		}


def get_matrix_maps(dimension):
	idx_to_coord = {i: idx for i, idx in enumerate([(i, j) for i in range(dimension) for j in range(dimension) if i <= j])}
	return len(idx_to_coord), idx_to_coord, {
		(i, j): [idx for idx in idx_to_coord if (
			(idx_to_coord[idx][0] == i and idx_to_coord[idx][1] == j) or
			(idx_to_coord[idx][0] == j and idx_to_coord[idx][1] == i)
		)][0]
		for i in range(dimension)
		for j in range(dimension)
	}


def get_factorization_element(q, idx_to_coord, i, j):
	possibles = [k for k, v in idx_to_coord.items() if v[0] == i and v[1] == j]
	if len(possibles) == 1:
		return q[possibles[0]]
	return 0


def q_is_feasible(q, p):
	for i in range(p.A.shape[0]):
		if numpy.dot(numpy.dot(p.A[i].T, q), p.A[i]) >= p.b[i]:
			return False
	return True


def find_feasible_starts(p):
	n = p.A.shape[1]
	q = numpy.zeros((n, n))
	delta = 1
	while delta > 1e-8:
		improved = True
		while improved:
			improved = False
			for i in range(n):
				delta_mat = numpy.pad([[delta]], pad_width=[[i, n-i-1], [i, n-i-1]], mode='constant')
				if not q_is_feasible(q + delta_mat, p):
					continue
				q += delta_mat
				improved = True
		delta /= 4
	yield numpy.array([
		[numpy.sqrt(xij) for xij in xi]
		for xi in q
	])


def get_starts(p, num_vars, idx_to_coord, bbar):
	if p.hot_start is not None:
		yield p.hot_start
	yield numpy.array([
		1.0 if idx_to_coord[i][0] == idx_to_coord[i][1] else 0.0
		for i in range(num_vars)
	])
	for starting_matrix in find_feasible_starts(Polyhedron(p.polyhedron.A, numpy.array([b*b/2 for b in bbar]))):
		yield numpy.array([
			starting_matrix[idx_to_coord[i][0], idx_to_coord[i][1]]
			for i in range(num_vars)
		])


def construct_ellipse(p, l_inverse, volume, k, bbar, hot_start=None):
	ellipse = Ellipse()
	ellipse.center = p.center
	if volume is not None:
		# This is no longer accurate.
		ellipse.volume = volume
	ellipse.hot_start = hot_start

	# l_inverse *= .8

	ellipse.l_inverse = l_inverse
	ellipse.l = numpy.linalg.inv(ellipse.l_inverse)

	ellipse.q_inverse = numpy.dot(ellipse.l_inverse.T, ellipse.l_inverse)
	ellipse.q = numpy.dot(ellipse.l.T, ellipse.l)

	ellipse.q_inverse = ellipse.q_inverse / (k * k)
	ellipse.q = k * k * ellipse.q

	ellipse.l = k * ellipse.l
	ellipse.l_inverse = ellipse.l_inverse / k

	ellipse.ds = []
	ellipse.lambdas = []

	if bbar is None:
		return ellipse

	for i in range(p.polyhedron.A.shape[0]):
		ar = p.polyhedron.A[i, :]
		br = bbar[i] / k
		qa = numpy.dot(ellipse.q_inverse, ar)
		lmbda = br / numpy.dot(ar, qa)
		d = lmbda * qa
		ellipse.lambdas.append(lmbda)
		ellipse.ds.append(d)
	return ellipse


def compute_maximal_ellipse(p):
	bbar = p.polyhedron.b - numpy.dot(p.polyhedron.A, p.center)
	k = 1.0 / min(abs(bbar))
	if k < 1e-10 or k > 1e8:
		print("========================================================")
		print("In maximize ellipse, scaling factor would be:")
		print(k)
		print("========================================================")
		return False, None
	bbar = k * bbar

	dimension = p.polyhedron.A.shape[1]
	idx_to_coord = {i: idx for i, idx in enumerate([(i, j) for i in range(dimension) for j in range(dimension) if i <= j])}
	num_vars = len(idx_to_coord)

	model = ConcreteModel()
	model.q = Var(range(num_vars))
	factorized_q = [
		[get_factorization_element(model.q, idx_to_coord, i, j) for j in range(dimension)]
		for i in range(dimension)
	]
	q_matrix = multiply(transpose(factorized_q), factorized_q)
	model.constraints = ConstraintList()

	for i in range(p.polyhedron.A.shape[0]):
		n = 1 / numpy.linalg.norm(numpy.array([xi for xi in p.polyhedron.A[i]] + [bbar[i]])) ** 2
		Ai = vector_to_matrix(p.polyhedron.A[i])
		if numpy.isnan(n):
			n = 1.0
		model.constraints.add(
			n * multiply(transpose(Ai), multiply(q_matrix, Ai))[0][0] <= bbar[i] * bbar[i] / 2 * n
		)

	#for i in range(num_vars):
	#	model.constraints.add(
	#		model.q[i] >= 0
	#	)

	# if p.include_point is not None:
	# 	if dimension != 2:
	# 		raise Exception('this is not supported')

	def objective_rule(m):
		det = 1.0
		for idx, coord in idx_to_coord.items():
			if coord[0] != coord[1]:
				continue
			det = det * m.q[idx]
		return det
	model.objective = Objective(rule=objective_rule, sense=maximize)

	maximum_l_inverse = None
	maximum_volume = None
	maximum_hotstart = None

	for start in get_starts(p, num_vars, idx_to_coord, bbar):
		solved = False
		for tolerance in [1e-16, 1e-10, 1e-8]:
			if solved:
				continue
			for i in range(num_vars):
				model.q[i].set_value(start[i])

			opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
			opt.options['tol'] = tolerance
			opt.options['warm_start_init_point'] = 'yes'
			try:
				result = opt.solve(model)
			except Exception as e:
				print('pyomo error', tolerance)
				# traceback.print_exc()
				# log_error(p)
				continue

			solved = True
			ok = result.solver.status == SolverStatus.ok
			if not ok:
				print("warning solver did not return ok")
				log_error(p)

			optimal = result.solver.termination_condition == TerminationCondition.optimal
			if not optimal:
				print("warning solver did not return optimal")
				log_error(p)

			if maximum_volume is not None and model.objective() <= maximum_volume:
				continue
			maximum_volume = model.objective()

			maximum_hotstart = [model.q[i].value for i in range(num_vars)]
			maximum_l_inverse = numpy.zeros((dimension, dimension))
			for i in range(num_vars):
				coord = idx_to_coord[i]
				maximum_l_inverse[coord[0], coord[1]] = model.q[i].value

	# model.pprint()

	if maximum_l_inverse is None:
		return False, False

	return True, construct_ellipse(p, maximum_l_inverse, -maximum_volume, k, bbar, maximum_hotstart)


def compute_maximal_ellipse_after_shift(params, l1):
	params2 = EllipseParams()
	params2.center = (params.center - l1.center) / l1.radius
	params2.polyhedron = params.polyhedron.shift(l1.center, l1.radius).normalize()
	# params2.include_point = (params.center - l1.center) / l1.radius if params.include_point is not None else None
	params2.tolerance = params.tolerance

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


'''

f(x + s) = f(x) + 
	D_0 f(x) (



'''