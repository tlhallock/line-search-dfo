
import numpy

from trust_region.dfo.lagrange_replacement_strategy import parse_replacement_policy
from trust_region.dfo.lagrange_replacement_strategy import ReplacementCheck


class Certification:
	def __init__(self):
		self.unshifted = None
		self.shifted = None
		self.indices = None
		self.original = None
		self.original_shifted = None
		self.poised = False
		self.forced_removal = None
		self.lmbda = None

	def fail(self):
		self.unshifted = None
		self.shifted = None
		self.indices = None
		self.original = None
		self.original_shifted = None
		self.poised = False
		self.lmbda = None

	def add_to_plot(self, plot_object):
		plot_object.add_points(points=self.original, label='original', color='b', marker='+', s=10)
		plot_object.add_points(points=self.unshifted, label='poised', color='k', marker='x', s=10)


def _test_v(v, basis, shifted):
	p = basis.basis_dimension
	n_points = shifted.shape[0]
	h = n_points + p
	error = numpy.linalg.norm(v[0:n_points, :] - basis.evaluate_to_matrix(shifted) * v[n_points:h, :]) / numpy.linalg.norm(v[0:n_points, :])
	if error > 1e-3:
		raise Exception("did not work: " + str(error))


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
	replacement_strategy_params,
	log_object
):
	checkers, options = parse_replacement_policy(replacement_strategy_params)

	p = basis.basis_dimension
	n_points = points.shape[0]
	h = n_points + p

	cert = Certification()
	cert.indices = [i for i in range(n_points)]
	cert.unshifted = points
	cert.shifted = trust_region.shift(points)
	cert.original = numpy.copy(cert.unshifted)
	cert.original_shifted = numpy.copy(cert.shifted)

	if not n_points == p:
		raise Exception("currently, have to have all points")

	# log_object['shifted'] = cert.shifted
	# log_object['shifted-vandermode'] = basis.evaluate_to_matrix(cert.shifted)

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

		current_point = cert.shifted[i]
		basis_coefficients = numpy.asarray(v[n_points:h, i]).flatten()
		check = ReplacementCheck(basis, trust_region, max_value, current_point, basis_coefficients)
		for checker in checkers:
			checker(check, options)
		if check.should_replace:
			# replace the row of V with new point
			cert.shifted[i] = check.new_point
			v[i] = numpy.dot(
				basis.evaluate_to_matrix(numpy.asarray([check.new_point])),
				v[n_points:h, :]
			)
			cert.indices[i] = -1
			_test_v(v, basis, cert.shifted)

		# perform LU
		v[:, i] = v[:, i] / v[i, i]
		for j in range(0, p):
			if i == j:
				continue
			v[:, j] = v[:, j] - v[i, j] * v[:, i]

	cert.unshifted = trust_region.unshift(cert.shifted)
	cert.lmbda = v[n_points:h]
	cert.poised = True

	# log_object['lambda'] = cert.lmbda

	_test_v(v, basis, cert.shifted)

	return cert
