
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


def _maximize_lagrange_quadratic(basis, trust_region, coefficients, initialization):
	model = ConcreteModel()
	model.dimension = range(basis.n)
	model.x = Var(model.dimension, bounds=(-1, 1))
	model.constraints = ConstraintList()
	if initialization is not None:
		for i in model.dimension:
			model.x[i].set_value(initialization[i])

	trust_region.add_shifted_pyomo_constraints(model)

	def objective_rule(m):
		return basis.to_pyomo_expression(m, coefficients)

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
	ret_value.objective = model.objective()
	ret_value.x = numpy.asarray([model.x[i]() for i in model.dimension])
	return ret_value


def maximize_lagrange_quadratic(basis, trust_region, coefficients):
	current = None
	for initialization in trust_region.sample_shifted_region(3):
		test = _maximize_lagrange_quadratic(basis, trust_region, [+c for c in coefficients], initialization=initialization)
		if current is None or abs(test.objective) > abs(current.objective):
			current = test
		test = _maximize_lagrange_quadratic(basis, trust_region, [-c for c in coefficients], initialization=initialization)
		if abs(test.objective) > abs(current.objective):
			current = test
	return current
