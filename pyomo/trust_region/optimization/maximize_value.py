
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy as np

from trust_region.util.plots import create_plot
from trust_region.util.bounds import Bounds

from trust_region.optimization.common import *


def get_pyomo_interpolate_objective(alpha, points, values):
	def objective_rule(model):
		distances = [
			(model.x[0] - point[0]) ** 2 + (model.x[1] - point[1]) ** 2
			for point in points
		]
		weights = [
			alpha / (d + alpha)
			for d in distances
		]
		weight_sum = 0
		for w in weights:
			weight_sum += w

		o = 0
		for i in range(len(weights)):
			o += values[i] * weights[i]
		return o / weight_sum
	return objective_rule


def _minimize_error(
	stencil_points,
	stencil_values
):
	model = ConcreteModel()
	dimension = 2
	model.dimension = range(dimension)
	model.x = Var(model.dimension)
	model.constraints = ConstraintList()
	model.constraints.add(
		model.x[0] <= 1
	)
	model.constraints.add(
		model.x[0] >= -1
	)
	model.constraints.add(
		model.x[1] <= 1
	)
	model.constraints.add(
		model.x[1] >= -1
	)

	model.objective = Objective(rule=get_pyomo_interpolate_objective(5, stencil_points, stencil_values), sense=minimize)
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		raise Exception("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		raise Exception("warning solver did not return optimal")

	# print(result)
	return np.array([model.x[i].value for i in range(dimension)])


def _minimize_other_polynomial(
	objective_rule
):
	model = ConcreteModel()
	dimension = 2
	model.dimension = range(dimension)
	model.x = Var(model.dimension)
	model.constraints = ConstraintList()
	model.constraints.add(
		model.x[0] <= 1
	)
	model.constraints.add(
		model.x[0] >= -1
	)
	model.constraints.add(
		model.x[1] <= 1
	)
	model.constraints.add(
		model.x[1] >= -1
	)

	model.objective = Objective(rule=objective_rule, sense=minimize)
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		raise Exception("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		raise Exception("warning solver did not return optimal")

	# print(result)
	return np.array([model.x[i].value for i in range(dimension)])


