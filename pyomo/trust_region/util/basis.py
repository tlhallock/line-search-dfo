import abc
import numpy


def parse_basis(basis_type, dimension):
	if basis_type == 'quadratic':
		return QuadraticBasis(dimension)
	elif basis_type == 'linear':
		return LinearBasis(dimension)
	else:
		raise Exception('unknown basis type: {}'.format(basis_type))


class Basis(metaclass=abc.ABCMeta):
	def __init__(self, n, basis_dimension):
		self.n = int(n)
		self.basis_dimension = int(basis_dimension)

	@abc.abstractmethod
	def evaluate_to_matrix(self, points):
		pass

	@abc.abstractmethod
	def to_pyomo_expression(self, model, coefficients):
		pass

	@abc.abstractmethod
	def evaluate_gradient(self, x, coefficients):
		pass


class QuadraticBasis(Basis):
	def __init__(self, n):
		Basis.__init__(self, n, 1 + n + n*(n+1)/2)

	def evaluate_to_matrix(self, points):
		n_points = points.shape[0]
		ret_val = numpy.zeros((n_points, self.basis_dimension))

		for i in range(n_points):
			idx = 0
			ret_val[i, idx] = 1.0
			idx += 1

			for j in range(self.n):
				ret_val[i, idx] = points[i, j]
				idx += 1

			for j in range(self.n):
				for k in range(j+1):
					ret_val[i, idx] = 0.5 * points[i, j] * points[i, k]
					idx += 1

		return ret_val

	def debug_evaluate(self, x, coefficients):
		if coefficients is None:
			coefficients = numpy.ones(6)
		return (
				1.0 * coefficients[0] +
				1.0 * coefficients[1] * x[0] +
				1.0 * coefficients[2] * x[1] +
				0.5 * coefficients[3] * x[0] * x[0] +
				0.5 * coefficients[4] * x[1] * x[0] +
				0.5 * coefficients[5] * x[1] * x[1]
		)

	def evaluate_gradient(self, x, coefficients):
		return numpy.array([
			1.0 * coefficients[1] +
			1.0 * coefficients[3] * x[0] +
			0.5 * coefficients[4] * x[1],
			1.0 * coefficients[2] +
			0.5 * coefficients[4] * x[0] +
			1.0 * coefficients[5] * x[1]
		])

	def to_pyomo_expression(self, model, coefficients):
		return (
				1.0 * coefficients[0] +
				1.0 * coefficients[1] * model.x[0] +
				1.0 * coefficients[2] * model.x[1] +
				0.5 * coefficients[3] * model.x[0] * model.x[0] +
				0.5 * coefficients[4] * model.x[1] * model.x[0] +
				0.5 * coefficients[5] * model.x[1] * model.x[1]
		)


class LinearBasis(Basis):
	def __init__(self, n):
		Basis.__init__(self, n, 1 + n)

	def evaluate_to_matrix(self, points):
		n_points = points.shape[0]
		ret_val = numpy.zeros((n_points, self.basis_dimension))

		for i in range(n_points):
			idx = 0
			ret_val[i, idx] = 1.0
			idx += 1

			for j in range(self.n):
				ret_val[i, idx] = points[i, j]
				idx += 1

		return ret_val

	def debug_evaluate(self, x, coefficients):
		return (
				1.0 * coefficients[0] +
				1.0 * coefficients[1] * x[0] +
				1.0 * coefficients[2] * x[1]
		)

	def evaluate_gradient(self, x, coefficients):
		return numpy.array([
			1.0 * coefficients[1],
			1.0 * coefficients[2]
		])

	def to_pyomo_expression(self, model, coefficients):
		return (
				1.0 * coefficients[0] +
				1.0 * coefficients[1] * model.x[0] +
				1.0 * coefficients[2] * model.x[1]
		)
