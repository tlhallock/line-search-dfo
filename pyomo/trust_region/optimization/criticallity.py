
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy

from trust_region.optimization.common import *


def _compute_projection(to_project, polyhedron):
	model = ConcreteModel()
	model.dimension = range(polyhedron.A.shape[1])
	model.x = Var(model.dimension)
	model.constraints = ConstraintList()

	polyhedron.add_to_pyomo(model)

	model.objective = Objective(
		expr=sum((model.x[c] - to_project[c]) * (model.x[c] - to_project[c]) for c in model.dimension),
		sense=minimize
	)
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")

	if not ok or not optimal:
		return False, None, None
	return True, numpy.sqrt(model.objective()), numpy.asarray([model.x[i]() for i in model.dimension])


def compute_projection(starting_point, polyhedron, l1):
	success, t, proj_x = _compute_projection(
		(starting_point - l1.center) / l1.radius,
		polyhedron.shift(l1.center, l1.radius)
	)
	if not success:
		return False, None
	return True,  proj_x * l1.radius + l1.center
