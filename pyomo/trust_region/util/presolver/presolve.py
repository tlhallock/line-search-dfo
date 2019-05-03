
import numpy as np
import trust_region.util.presolver.expressions as exprs
import sys
from trust_region.util.utils import write_json
from scipy.optimize import linprog


# class SqpState:
# 	def __init__(self):
# 		self.A = None
# 		self.b = None
#
# 		self.P = None
# 		self.q = None
# 		self.r = None
#
# 		self.active_constraints = []
#
# 	def active_equalities(self):
# 		return (
# 			np.array([self.A[i] for i in self.active_constraints]),
# 			np.array([self.b[i] for i in self.active_constraints]),
# 		)
#
# 	def construct_kkt_system(self):
# 		active_A, active_b = self.active_equalities()
# 		m = active_A.shape[0]
# 		n = active_A.shape[1]
# 		return np.vstack([
# 			np.hstack([self.Q, active_A.T]),
# 			np.hstack([active_A, np.zeros((m, n))])
# 		]), np.vstack([-self.q, active_b])


class ClassicModel:
	def __init__(self):
		self.x = None
		self.r = None

		self.c = None
		self.A = None

		self.f = None
		self.g = None

	def to_json(self):
		return {
			'x': self.x,
			'r': self.r,
			'c': self.c,
			'A': self.A,
			'f': self.f,
			'g': self.g
		}


class Params:
	def __init__(self, variable, f, c, x0, r0, tolerance=1e-12):
		self.tolerance = tolerance
		self.variable = variable
		self.f_func = f
		self.c_func = c
		self.r0 = 1.0
		self.x0 = x0
		self.r0 = r0

		self.A_func = exprs.simplify(c.jacobian(variable))
		self.g_func = exprs.simplify(f.gradient(variable))

		self.restoration_f = exprs.simplify(exprs.Sum([
			exprs.Power(exprs.PositivePart(component), exprs.Constant(2))
			for component in c.components
		]))
		self.restoration_g = exprs.simplify(self.restoration_f.gradient(variable))

	def pretty_print(self):
		return "minimize\n" + self.f_func.pretty_print() + "\nsubject to\n" + self.c_func.pretty_print()


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


def construct_restoration_model(params, x, r):
	model = ClassicModel()
	model.x = x
	model.r = r
	variables = {'x': x}
	model.c = np.zeros(0)
	model.A = np.zeros((0, len(x)))
	model.f = params.restoration_f.evaluate(variables)
	model.g = params.restoration_g.evaluate(variables)
	return model


def solve_trust_region_subproblem(model, center, radius):
	# print('==========================================')
	# print('minimize', model.g)
	# print('subject to')
	# print(model.A)
	# print(-model.c)
	# print([[-radius, + radius] for _ in center])
	# print('==========================================')
	try:
		result = linprog(
			c=model.g,
			A_ub=model.A,
			b_ub=-model.c,
			bounds=[[-radius, radius] for _ in center]
		)
	except ValueError:
		return {
			'infeasible': True
		}
	if not result.success and result.message == 'Optimization failed. Unable to find a feasible starting point.':
		return {
			'infeasible': True
		}

	return {
		'trial-value': model.f + np.dot(model.g, result.x),
		'trial-point': center + result.x,
		'success': result.success,  # result.message == 'Optimization terminated successfully'
		'infeasible': False
	}


def run_iteration(model, function, center, radius, tolerance):
	solution = solve_trust_region_subproblem(model, center, radius)

	if solution['infeasible']:
		return {
			'status': 'infeasible',
			'radius': radius,
			'center': center
		}

	dist = np.linalg.norm(solution['trial-point'] - model.x)
	if dist < tolerance:
		if (model.c <= tolerance).all():
			return {
				'status': 'critical',
				'radius': radius,
				'center': center
			}

	trial_objective = function.evaluate({'x': solution['trial-point']})
	if abs(model.f - solution['trial-value']) < tolerance:
		rho = 0.0
	else:
		rho = (model.f - trial_objective) / (model.f - solution['trial-value'])
	# print('-------------------------------------------')
	# print('\trho', rho)
	# print('\ttrial point', solution['trial-point'])
	# print('\tcurrent', model.f)
	# print('\tpredicted', solution['trial-value'])
	# print('\tactual', trial_objective)
	# print('-------------------------------------------')

	if np.isnan(rho) or np.isinf(rho):
		rho = 0.0

	if rho < 0.1:
		return {
			'status': 'rejected',
			'radius': radius * 0.1,
			'center': center
		}

	return {
		'status': 'accepted',
		'radius': radius * (0.1 if dist < radius else (1.5 if rho > 0.8 else 1.0)),
		'center': solution['trial-point']
	}


def solve(params):
	center = params.x0
	radius = params.r0
	count = 0
	while True:
		count += 1
		if count > 10000:
			raise Exception('maximum iterations hit')

		model = construct_model(params, center, radius)
		print('==========================================')
		print('center', center)
		print('radius', radius)
		print('objective', model.f)
		print('constraints', model.c)
		print('==========================================')

		result = run_iteration(model, params.f_func, center, radius, params.tolerance)
		# print(result)

		# if (model.c <= 0).all() and result['status'] == 'infeasible':
		# 	run_iteration(model, params.f_func, center, radius, params.tolerance)

		if result['status'] == 'infeasible':
			model = construct_restoration_model(params, center, radius)
			result = run_iteration(model, params.restoration_f, center, radius, params.tolerance)

		if result['status'] == 'critical':
			return {
				'success': True,
				'minimizer': center,
				'value': model.f,
				'constraints': model.c
			}

		if result['status'] in ['rejected', 'accepted']:
			center = result['center']
			radius = result['radius']

#
# var = exprs.create_variable_array("x", 2)
# params = Params(
# 	variable=var,
# 	f=exprs.Sum([
# 		exprs.Power(var.get(0), exprs.Constant(2)),
# 		exprs.Power(
# 			exprs.Sum([
# 				var.get(1),
# 				exprs.Constant(-1.0)
# 			]),
# 			exprs.Constant(2)
# 		)
# 	]),
# 	c=exprs.Vector([
# 		exprs.Negate(var.get(0)),
# 		exprs.Sum([
# 			var.get(1),
# 			exprs.Negate(var.get(0))
# 		])
# 	]),
# 	x0=np.array([1, 10]),
# 	r0=1
# )
#
# print(params.restoration_f.pretty_print())
# print(params.restoration_g.pretty_print())
#
# solve(params)
