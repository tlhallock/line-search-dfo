
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *


class LagrangeMaximization:
	def __init__(self):
		self.success = False
		self.x = None
		self.objective = None


def _maximize_lagrange_quadratic(coefficients, initialization):
	model = ConcreteModel()
	model.dimension = range(2) # dimension...
	model.x = Var(model.dimension, initialize=initialization, bounds=(-1, 1))
	model.constraints = ConstraintList()
	model.constraints.add(
		# Trust region
		sum(model.x[i] * model.x[i] for i in model.dimension) <= 1.0
	)

	def objective_rule(m):
		return (
			1.0 * coefficients[0] +
			1.0 * coefficients[1] * m.x[0] +
			1.0 * coefficients[2] * m.x[1] +
			0.5 * coefficients[3] * m.x[0] * m.x[0] +
			0.5 * coefficients[4] * m.x[1] * m.x[0] +
			0.5 * coefficients[5] * m.x[1] * m.x[1]
		)
	model.objective = Objective(rule=objective_rule, sense=maximize)
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")

	ret_value = LagrangeMaximization()
	ret_value.success = ok and optimal
	ret_value.objective  = model.objective()
	ret_value.x = numpy.asarray([model.x[i]() for i in model.dimension])
	return ret_value


def maximize_lagrange_quadratic(coefficients):
	current = None
	for initialization in (0.0, -0.1, .1):
		test = _maximize_lagrange_quadratic([+c for c in coefficients], initialization=initialization)
		if current is None or abs(test.objective) > abs(current.objective):
			current = test
		test = _maximize_lagrange_quadratic([-c for c in coefficients], initialization=initialization)
		if abs(test.objective) > abs(current.objective):
			current = test
	return current
