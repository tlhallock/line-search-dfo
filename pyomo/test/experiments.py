import numpy as np
from scipy import integrate

from trust_region.util.plots import create_plot
from trust_region.util.bounds import Bounds
from trust_region.optimization.maximize_value import _minimize_error, _minimize_other_polynomial

from trust_region.util.simple_basis import BASIS_POLYNOMIALS_2D

from trust_region.util.nullspace import nullspace

np.set_printoptions(linewidth=255)
'''

increase in accuracy
	constants about current points

probability of being out of bounds

cost of evaluating a point


two stage






'''



class Model:
	def __init__(self, points, values, num_higher_terms=2):
		self.num_higher_terms = num_higher_terms
		self.npoints = points.shape[0]
		self.basis_polys = BASIS_POLYNOMIALS_2D[:self.npoints]
		self.higher_order_polys = BASIS_POLYNOMIALS_2D[:self.npoints + self.num_higher_terms]
		self.vandermode = np.array([[poly['func'](point) for poly in self.basis_polys] for point in points])
		self.inverse = np.linalg.inv(self.vandermode)
		self.values = values
		self.nspace = nullspace(
			np.array([[poly['func'](point) for poly in self.higher_order_polys] for point in points])
		)

	def evaluate(self, x):
		return np.dot(self.values, np.dot(np.array([poly['func'](x) for poly in self.basis_polys]), self.inverse))

	def evaluate_objective(self, model):
		return self.evaluate(model.x)

	def evaluate_higher_terms(self, x, theta):
		return np.dot(
			np.hstack([self.values, theta]),
			np.dot(
				np.array([poly['func'](x) for poly in self.higher_order_polys]),
				np.hstack([np.vstack([self.inverse, np.zeros((self.num_higher_terms, self.npoints))]), self.nspace])
			)
		)

	def higher_order_term_objective(self, theta):
		new_coefficients = np.hstack([self.values, theta])
		new_inverse = np.hstack([np.vstack([self.inverse, np.zeros((self.num_higher_terms, self.npoints))]), self.nspace])

		def objective_rule(model):
			return np.dot(
				new_coefficients,
				np.dot(
					np.array([poly['func'](model.x) for poly in self.higher_order_polys]),
					new_inverse
				)
			)
		return objective_rule

	def get_derivative(self, idx, order, x):
		if order == 0:
			return np.dot(np.array([poly['func'](x) for poly in self.basis_polys]), self.inverse[:, idx])
		elif order == 1:
			return np.dot(np.array([poly['grad'](x) for poly in self.basis_polys]).T, self.inverse[:, idx])
		elif order == 2:
			raise np.dot(np.array([poly['hess'](x) for poly in self.basis_polys]).T, self.inverse[:, idx])
		else:
			raise Exception('derivative too high')



def objective(x):
	a = 0.01
	b = 1
	return (a - x[0]) ** 2 + b*(x[1] - x[0] ** 2) ** 2


def lip_error(x, points):
	return min(np.linalg.norm(points - x, axis=1))


def get_difference(original_points, trial_point):
	new_points = np.vstack([points, trial_point])
	difference = integrate.nquad(
		lambda x, y: (
			lip_error(np.array([x, y]), new_points) - lip_error(np.array([x, y]), original_points)
		),
		[[-1, 1], [-1, 1]],
		opts={'epsabs': 1e-2},
		full_output=True
	)
	return difference[0]


def zero_order_interpolate(alpha, points, values, x):
	weights = alpha / (np.linalg.norm(points - x, axis=1) ** 2 + alpha)
	weights /= sum(weights)
	return np.dot(weights, values)


def probability_distribution(beta, points, x):
	distances = np.linalg.norm(points - x, axis=1) ** 2
	return sum(
		1/(1 + distances * beta ** 2)
	) / (points.shape[0] * np.pi / beta)


def get_ciarlet_error(model, x, r, d, v_bounds):
	# \| \nabla^r f(x) - \nabla^r m(x)  \| \le 1 / (d)! v_{d-1} \sigma_i \|y^i - x\|^d \|\nabla^r l_i(x)\|
	ret = 1.0
	for i in range(1, d+2):
		ret /= i
	ret *= v_bounds[d]
	s = 0.0
	for i in range(points.shape[0]):
		s += np.linalg.norm(points[i] - x) ** (d+1) * np.linalg.norm(model.get_derivative(i, r, x))
	return ret * s


points = np.array([
	[-0.25, .75],
	[.75, .75],
	[0.5, -0.75]
])

for iteration in range(5):
	print('\t\t\tITERATION', iteration)

	model = Model(points, np.array([objective(point) for point in points]))


	other_minimums = np.array([
		[xi for xi in _minimize_other_polynomial(model.higher_order_term_objective(theta))]
		for theta in np.random.normal(0, 2, (100, 2))
	])
	current_minimum = _minimize_other_polynomial(model.evaluate_objective)

	NUM_PER_DIMENSION = int(np.sqrt(30))
	stencil = np.array([
		[x, y]
		for x in np.linspace(-1, 1, NUM_PER_DIMENSION)
		for y in np.linspace(-1, 1, NUM_PER_DIMENSION)
	])

	print('integrating')
	stencil_values = np.array([get_difference(points, x) for x in stencil])
	print('done')

	L = 1

	bounds = Bounds()
	bounds.extend(np.array([L, L]))
	bounds.extend(np.array([-L, -L]))
	p = create_plot('something', 'experimenting/something_' + str(iteration) + '.png', bounds)

	p.add_points(other_minimums, label='other minimums', color='b')
	p.add_points(np.expand_dims(current_minimum, axis=0), label='current minimum', color='g')

	p.add_points(points, label='points')
	p.add_contour(lambda x: lip_error(x, points), label='lip error', color='r')
	p.add_contour(lambda x: get_ciarlet_error(model, x, 1, 0, [1]), label='error in gradient', color='c')

	p.add_contour(lambda x: zero_order_interpolate(5, stencil, stencil_values, x), label='decrease_in_error', color='g')

	p.add_points(stencil, label='stencil', color='y')
	p.add_contour(lambda x: model.evaluate(x), label='model', color='y')
	p.add_contour(lambda x: probability_distribution(0.1, other_minimums, x), label='prob distribution', color='b')
	p.add_contour(objective, label='objective')

	p.save()

	print('maximizing')
	new_point = _minimize_error(stencil, stencil_values)
	print(new_point)
	print('done')

	points = np.vstack([points, new_point])
