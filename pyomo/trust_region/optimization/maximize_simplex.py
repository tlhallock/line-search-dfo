
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.util.plots import create_plot
from trust_region.util.bounds import Bounds

from trust_region.optimization.common import *


def _quadratic_constraint(Q, b, c, x, p):
	return (
		c +
		b[0] * x[p, 0] +
		b[1] * x[p, 1] +
		0.5 * Q[0, 0] * x[p, 0] * x[p, 0] +
		0.5 * (Q[1, 0] + Q[0, 1]) * x[p, 1] * x[p, 0] +
		0.5 * Q[1, 1] * x[p, 1] * x[p, 1]
	) <= 0


def _determinant(matrix):
	if len(matrix) == 1:
		return matrix[0][0]

	ret = 0
	sgn = 1
	for j in range(len(matrix[0])):
		ret = ret + sgn * matrix[0][j] * _determinant(
			[
				[
					matrix[ii][jj]
					for jj in range(len(matrix))
					if jj != j
				]
				for ii in range(1, len(matrix))
			]
		)
		sgn = -sgn

	return ret


def _volume_of_simplex(points, dimension, num_points):
	return _determinant([
		[points[i, j] - points[0, j] for j in range(dimension)]
		for i in range(1, num_points)
	])


def _maximize_simplex(
		x0,
		Qs,
		bs,
		cs
):
	model = ConcreteModel()
	dimension = len(x0)
	num_points = len(x0) + 1
	model.dimension = range(dimension)
	print(dimension)
	model.points = Var(range(num_points), model.dimension)
	for i in range(num_points):
		for j in range(dimension):
			model.points[i, j].set_value(
				numpy.random.random() / 100
			)

	model.constraints = ConstraintList()

	for Q, b, c in zip(Qs, bs, cs):
		for p in range(num_points):
			model.constraints.add(
				_quadratic_constraint(Q, b, c, model.points, p)
			)

	# Needs to be shifted first
	def objective_rule(m):
		return _volume_of_simplex(m.points, dimension, num_points)

	model.objective = Objective(rule=objective_rule, sense=maximize)
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		raise Exception("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		raise Exception("warning solver did not return optimal")

	print(result)
	return numpy.array([
		[
			model.points[i, j].value
			for j in range(dimension)
		]
		for i in range(num_points)
	])


Qs = [
	numpy.array([
		[10, 5],
		[0, 1]
	]),
	numpy.array([
		[1, 0],
		[0, 10]
	])
]
bs = [
	numpy.array([1, 0]),
	numpy.array([0, 0])
]
cs = [-10, -5]
#
# print(_determinant(
# 	[
# 		[2, 3],
# 		[4, 7]
# 	]
# ))

points = _maximize_simplex(
	numpy.array([0, 0]),
	Qs,
	bs,
	cs
)


bounds = Bounds()
bounds.extend(numpy.array([10, 10]))
bounds.extend(numpy.array([-10, -10]))
p = create_plot('a simplex', 'simplex.png', bounds)
for Q, b, c in zip(Qs, bs, cs):
	p.add_contour(
		lambda x: (0.5 * numpy.dot(numpy.dot(x, Q), x) + numpy.dot(b, x) + c),
		label='some quadratic'
	)
p.add_points(points, label='simplex')
p.save()