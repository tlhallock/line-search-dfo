
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *


class TrustRegionSubProblem:
	def __init__(self):
		self.success = False
		self.predicted_objective_value = None
		self.trial_point = None


def _solve_trust_region_subproblem(
		objective_basis,
		objective_coefficients,
		outer_trust_region,
		trust_region,
		initialization
):
	model = ConcreteModel()
	model.dimension = range(objective_basis.n)
	model.x = Var(model.dimension)
	model.constraints = ConstraintList()

	outer_trust_region.add_unshifted_pyomo_constraints(model)
	trust_region.add_unshifted_pyomo_constraints(model)

	# Needs to be shifted first
	def objective_rule(m):
		return objective_basis.to_pyomo_expression(model=m, coefficients=objective_coefficients)

	model.objective = Objective(rule=objective_rule, sense=minimize)
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		raise Exception("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		raise Exception("warning solver did not return optimal")

	ret_value = TrustRegionSubProblem()
	ret_value.success = ok and optimal
	ret_value.predicted_objective_value = model.objective()
	ret_value.trial_point = numpy.asarray([model.x[i]() for i in model.dimension])
	return ret_value


def solve_trust_region_subproblem(
		objective_basis,
		objective_coefficients,
		model_center,
		outer_trust_region,
		trust_region
):
	return _solve_trust_region_subproblem(
		objective_basis,
		objective_coefficients,
		outer_trust_region,
		trust_region,
		model_center
	)
	# if current is None or abs(test['objective']) > abs(current['objective']):
	#	current = test
