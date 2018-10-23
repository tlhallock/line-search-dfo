import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *

from trust_region.dfo.trust_region.polyhedral_trust_region import _add_constraints_to_pyomo


def minimize_other_polynomial(
		n,
		objective_basis,
		shifted_polyhedron,
		initialization,
		sample_coefficients
):
	model = ConcreteModel()
	model.dimension = range(n)
	model.x = Var(model.dimension)
	model.constraints = ConstraintList()

	_add_constraints_to_pyomo(
		model,
		shifted_polyhedron[0],
		shifted_polyhedron[1]
	)

	model.objective = Objective(
		expr=objective_basis.to_pyomo_expression(model, sample_coefficients),
		sense=minimize
	)
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	if result.solver.status != SolverStatus.ok or result.solver.termination_condition != TerminationCondition.optimal:
		return None
	return numpy.asarray([model.x[i]() for i in model.dimension])

