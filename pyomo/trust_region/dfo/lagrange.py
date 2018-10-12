
import numpy
import matplotlib.pyplot as plt
from trust_region.optimization.maximize_lagrange import maximize_lagrange_quadratic


class LagrangeParams:
	def __init__(self):
		self.improve_with_new_points = True
		self.xsi = 0.00001
		self.far_radius = 1.5


class Certification:
	def __init__(self):
		self.unshifted = None
		self.shifted = None
		self.indices = None
		self.original = None
		self.original_shifted = None
		self.poised = False
		self.forced_removal = None

	def fail(self):
		self.unshifted = None
		self.shifted = None
		self.indices = None
		self.original = None
		self.original_shifted = None
		self.poised = False
		# lambdas = numpy.empty(n_points)

	def add_to_plot(self, plot_object):
		plot_object.add_points(points=self.original, label='original', color='b', marker='+', s=10)
		plot_object.add_points(points=self.unshifted, label='poised', color='k', marker='x', s=10)


def _test_v(v, basis, shifted):
	p = basis.basis_dimension
	n_points = shifted.shape[0]
	h = n_points + p
	if numpy.linalg.norm(v[0:n_points, :] - basis.evaluate_to_matrix(shifted) * v[n_points:h, :]) > 1e-3:
		raise Exception("did not work")


def _get_max_index(vec):
	idx = 0
	val = vec[0, 0]
	for i in range(0, vec.shape[0]):
		new_val = vec[i, 0]
		if new_val > val:
			idx = i
			val = new_val
	return val, idx


def _replace_row(
		basis,
		certification,
		i,
		new_sample_point,
		n_points,
		h,
		v
):
	certification.shifted[i] = new_sample_point
	v[i] = numpy.dot(
		basis.evaluate_to_matrix(numpy.asarray([new_sample_point])),
		v[n_points:h, :]
	)
	certification.indices[i] = -1
	_test_v(v, basis, certification.shifted)
	return _get_max_index(abs(v[i:n_points, i]))


def compute_lagrange_polynomials(
		basis,
		trust_region,
		points,
		context,
		lagrange_params
):
	p = basis.basis_dimension
	n_points = points.shape[0]
	h = n_points + p

	cert = Certification()
	cert.indices = [i for i in range(n_points)]
	cert.unshifted = points
	cert.shifted = trust_region.shift(points)
	cert.original = numpy.copy(cert.unshifted)
	cert.original_shifted = numpy.copy(cert.shifted)
	cert.forced_removal = [
		lagrange_params.far_radius is not None and numpy.linalg.norm(cert.shifted[i]) > lagrange_params.far_radius
		for i in range(points.shape[0])
	]

	if not n_points == p:
		raise Exception("currently, have to have all points")

	v = numpy.bmat([
		[basis.evaluate_to_matrix(cert.shifted)],
		[numpy.eye(p)]
	])

	for i in range(0, p):
		_test_v(v, basis, cert.shifted)

		# Get maximum value in matrix
		max_value, max_index = _get_max_index(abs(v[i:n_points, i]))

		# perform pivot
		if not max_index == 0:
			other_idx = max_index + i
			v[[i, other_idx], :] = v[[other_idx, i], :]
			cert.shifted[[i, other_idx], :] = cert.shifted[[other_idx, i], :]

			tmp = cert.indices[i]
			cert.indices[i] = cert.indices[other_idx]
			cert.indices[other_idx] = tmp

			tmp = cert.forced_removal[i]
			cert.forced_removal[i] = cert.forced_removal[other_idx]
			cert.forced_removal[other_idx] = tmp

		# Check the poisedness
		if lagrange_params.improve_with_new_points and (
			cert.forced_removal[i] or
			max_value < lagrange_params.xsi
		):
			# If still not poised, Then check for new points
			coefficients = numpy.asarray(v[n_points:h, i]).flatten()
			maximization_result = maximize_lagrange_quadratic(coefficients)

			# replace the row of V with new point
			cert.shifted[i] = maximization_result.x
			v[i] = numpy.dot(
				basis.evaluate_to_matrix(numpy.asarray([maximization_result.x])),
				v[n_points:h, :]
			)
			cert.indices[i] = -1
			cert.forced_removal[i] = False
			_test_v(v, basis, cert.shifted)

			max_value = abs(v[i, i])

		if max_value < lagrange_params.xsi and lagrange_params.improve_with_new_points:
			print('This is a problem')

		# perform LU
		v[:, i] = v[:, i] / v[i, i]
		for j in range(0, p):
			if i == j:
				continue
			v[:, j] = v[:, j] - v[i, j] * v[:, i]

	cert.unshifted = trust_region.unshift(cert.shifted)
	cert.lmbda = v[n_points:h]
	cert.poised = True

	_test_v(v, basis, cert.shifted)

	return cert
