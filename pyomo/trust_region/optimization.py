
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy


def _maximize_lagrange_quadratic(coefficients):
	model = ConcreteModel()
	model.dimension = range(2)
	model.x = Var(model.dimension, initialize=0.5)
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
	opt = SolverFactory('ipopt')
	result = opt.solve(model)
	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")
	return {
		'success': ok and optimal,
		'objective': model.objective(),
		'x':  numpy.asarray([model.x[i]() for i in model.dimension])
	}


def maximize_lagrange_quadratic(coefficients):
	r1 = _maximize_lagrange_quadratic([+c for c in coefficients])
	r2 = _maximize_lagrange_quadratic([-c for c in coefficients])
	if abs(r2['objective']) > abs(r1['objective']):
		return r2
	else:
		return r1


print(maximize_lagrange_quadratic([0, 1, 1, 0, 0, 0]))
