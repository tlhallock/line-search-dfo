
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *

from trust_region.util.bounds import Bounds
from trust_region.util.plots import create_plot


class TrustRegionSubProblem:
	def __init__(self):
		self.success = False
		self.predicted_objective_value = None
		self.trial_point = None

# cntr = 0


def solve_trust_region_subproblem(
		objective_basis,
		objective_coefficients,
		trust_region,
		model_center,
		buffer
):
	model = ConcreteModel()
	model.dimension = range(objective_basis.n)
	model.x = Var(model.dimension)
	model.constraints = ConstraintList()

	trust_region.add_shifted_pyomo_constraints(model)
	if buffer is not None:
		buffer.add_to_pyomo(model)

	# Needs to be shifted first
	def objective_rule(m):
		return objective_basis.to_pyomo_expression(
			model=m,
			coefficients=objective_coefficients
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

	shifted_solution = numpy.asarray([model.x[i]() for i in model.dimension])

	# global cntr
	# bounds = Bounds()
	# bounds.extend(numpy.array([-2, -2]))
	# bounds.extend(numpy.array([6, 6]))
	# p = create_plot('tr', 'images/higher_dimension/trust_region_subproblem_' + str(cntr) + '.png', bounds)
	# p.add_polyhedron(buffer, label='buffer', color='r')
	# p.add_polyhedron(constraints, label='buffer', color='b')
	# trust_region.add_to_plot(p)
	# p.save()
	#
	# cntr += 1


	ret_value = TrustRegionSubProblem()
	ret_value.success = ok and optimal
	ret_value.predicted_objective_value = model.objective()
	ret_value.trial_point = trust_region.unshift_row(shifted_solution)
	return ret_value
