
import numpy as np
import trust_region.util.presolver.expressions as exprs
from scipy.optimize import linprog


class SqpState:
	def __init__(self):
		self.A = None
		self.b = None

		self.P = None
		self.q = None
		self.r = None

		self.active_constraints = []

	def active_equalities(self):
		return (
			np.array([self.A[i] for i in self.active_constraints]),
			np.array([self.b[i] for i in self.active_constraints]),
		)

	def construct_kkt_system(self):
		active_A, active_b = self.active_equalities()
		m = active_A.shape[0]
		n = active_A.shape[1]
		return np.vstack([
			np.hstack([self.Q, active_A.T]),
			np.hstack([active_A, np.zeros((m, n))])
		]), np.vstack([-self.q, active_b])


class ClassicModel:
	def __init__(self):
		self.x = None
		self.r = None

		self.c = None
		self.A = None

		self.f = None
		self.g = None


class Params:
	def __init__(self, variable, f, c, x0, r0):
		self.variable = variable
		self.f_func = f
		self.c_func = c
		self.r0 = 1.0
		self.x0 = x0
		self.r0 = r0

		self.A_func = exprs.simplify(c.jacobian(variable))
		self.g_func = exprs.simplify(f.gradient(variable))

		print(self.g_func.pretty_print())
		print(self.A_func.pretty_print())

		self.restoration_objective = exprs.SumExpression([
			exprs.PowerExpression(exprs.PositivePart(component), exprs.ConstantExpression(2))
			for component in c.components
		])


class State:
	def __init__(self, params):
		self.params = params
		self.radius = params.r0
		self.center = params.x0


def construct_model(params, x, r):
	model = ClassicModel()
	model.x = x
	model.r = r
	variables = {'x': x}
	model.c = params.c_func.evaluate(variables)
	model.A = params.A_func.evaluate(variables)
	model.f = params.f_func.evaluate(variables)
	model.g = params.g_func.evaluate(variables)
	return model


def is_critical(gradient, A, radius):
	return False


def solve_trust_region_subproblem(model, center, radius):
	result = linprog(
		c=model.g,
		A_ub=model.A,
		b_ub=np.zeros(model.A.shape[0]),
		bounds=[[c - radius,  c + radius] for c in center]
	)
	return {
		'trial-value': model.f + np.dot(model.g, result.x - center),
		'trial-point': result.x,
		'success': result.success # result.message == 'Optimization terminated successfully'
	}


def run_iteration(state):
	m_k = construct_model(state.params, state.center, state.radius)
	print(m_k.x, m_k.r, m_k.f)

	if is_critical(m_k.g, m_k.A, state.radius):
		return True

	solution = solve_trust_region_subproblem(m_k, state.center, state.radius)
	trial_objective = state.params.f_func.evaluate({'x': solution['trial-point']})
	# print('f(s)', trial_objective)

	rho = (m_k.f - trial_objective) / (m_k.f - solution['trial-value'])
	print('rho', rho)

	if rho < 0.1:
		state.radius *= 0.5
	else:
		if rho > 0.8:
			state.radius *= 1.2
		state.center = solution['trial-point']


def solve(params):
	state = State(params)
	while True:
		if run_iteration(state):
			break


	# construct model
	#
	pass


var = exprs.create_variable_array("x", 2)
params = Params(
	variable=var,
	f=exprs.SumExpression([
		exprs.PowerExpression(var.get(0), exprs.ConstantExpression(2)),
		exprs.PowerExpression(
			exprs.SumExpression([
				var.get(1),
				exprs.ConstantExpression(-1.0)
			]),
			exprs.ConstantExpression(2)
		)
	]),
	c=exprs.VectorExpression([
		exprs.NegateExpression(var.get(0)),
		exprs.SumExpression([
			var.get(1),
			exprs.NegateExpression(var.get(0))
		])
	]),
	x0=np.array([10, 1]),
	r0=1
)


solve(params)
