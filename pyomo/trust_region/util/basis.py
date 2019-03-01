import abc
import numpy
import itertools
import math


def parse_basis(basis_type, dimension):
	if basis_type == 'quadratic':
		return PolynomialBasis(dimension, 2)
	elif basis_type == 'linear':
		return PolynomialBasis(dimension, 1)
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


class Monomial:
	def __init__(self, powers, coefficient, degree):
		self.powers = powers
		self.coefficient = coefficient
		self.degree = degree

	def differentiate(self, idx):
		if self.powers[idx] == 0:
			return Monomial(
				[0 for _ in range(len(self.powers))],
				0.0,
				0
			)
		return Monomial(
			[p if i != idx else p-1 for i, p in enumerate(self.powers)],
			self.coefficient * self.powers[idx],
			-1
		)

	def to_pyomo(self, model):
		ret = self.coefficient
		for comp, p in enumerate(self.powers):
			for i in range(p):
				ret = ret * model.x[comp]
		return ret

	def evaluate(self, point):
		ret = self.coefficient
		for comp, p in enumerate(self.powers):
			ret *= point[comp] ** p
		return ret

	def pretty_print(self):
		products = "*".join([
			"x[" + str(i) + "]" + (" ** " + str(p) if p > 1 else "")
			for i, p in enumerate(self.powers)
			if p > 0
		])
		return str(self.coefficient) + ("*" + products if products != "" else "")


class PolynomialBasis:
	def __init__(self,  dimension, order):
		self.dimension = dimension
		self.order = order
		self.monomials = [
			Monomial(
				[
					len([k for k in term if k == i])
					for i in range(dimension)
				],
				1.0 / math.factorial(degree),
				degree
			)
			for degree in range(order + 1)
			for term in itertools.combinations_with_replacement(list(range(dimension)), degree)
		]

	@property
	def basis_dimension(self):
		return len(self.monomials)

	@property
	def n(self):
		return self.dimension

	def evaluate_to_matrix(self, points):
		return numpy.array([
			[
				monomial.evaluate(points[i])
				for monomial in self.monomials
			]
			for i in range(points.shape[0])
		])

	def to_pyomo_expression(self, model, coefficients):
		ret = 0
		for i, monomial in enumerate(self.monomials):
			ret = ret + coefficients[i] * monomial.to_pyomo(model)
		return ret

	def debug_evaluate(self, x, coefficients):
		ret = 0
		for i, monomial in enumerate(self.monomials):
			ret += coefficients[i] * monomial.evaluate(x)
		return ret

	def evaluate_gradient(self, x, coefficients):
		return numpy.array([
			sum([
				coefficients[j] * monomial.differentiate(i).evaluate(x)
				for j, monomial in enumerate(self.monomials)
			])
			for i in range(self.dimension)
		])
