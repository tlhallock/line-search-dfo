
import numpy

class LagrangeParams:
	def  __init__(self):
		self.improve_with_new_points = True
		self.xsi = 0.00001

class Certification:
	def __init__(self):
		self.unshifted = None
		self.shifted = None
		self.poised = False
		self.shifted = None

	def fail(self):
		self.poised = False
		self.shifted = None
		self.unshifted = None
		# lambdas = numpy.empty(n_points)

	def plot(self, filename, center, radius):
		#fig = plt.figure()
		#fig.set_size_inches(sys_utils.get_plot_size(), sys_utils.get_plot_size())
		#ax1 = fig.add_subplot(111)
		#ax1.add_artist(plt.Circle(center, radius, color='g', fill=False))
		#ax1.scatter(self.original[:, 0], self.original[:, 1], s=10, c='b', marker="+", label='original')
		#ax1.scatter(self.unshifted[:, 0], self.unshifted[:, 1], s=10, c='r', marker="x", label='poised')

		# ax1.axis([center[0] - 2 * radius, center[0] + 2 * radius, center[1] - 2 * radius, center[1] + 2 * radius])

		#if self.Lambda is not None:
		#	lambdaStr = "Lambda=" + str(max(self.Lambda))
		#	ax1.text(center[0], center[1], lambdaStr)

		#plt.legend(loc='lower left')
		#fig.savefig(filename)
		#plt.close()


def _test_v(V, basis, shifted):
	pass


def _swap_rows(mat, idx1, idx2):
	mat[[idx1, idx2], :] = mat[[idx2, idx1], :]


def _get_max_index(max):
	idx = 0
	val = max[0,0]
	for i in range(0, max.shape[0]):
		new_val = max[i, 0]
		if new_val > val:
			idx = i
			val = new_val
	return val, idx


def _replace(cert, i, new_value, n_points, h, V, b):
	cert.shifted[i] = new_value
	V[i] = dot(b.evaluateRowToRow(new_value), V[n_points:h, :])
	cert.indices[i] = -1
	_test_v(V, b, cert.shifted)
	return _get_max_index(abs(V[i:n_points, i]))


def computeLagrangePolynomials(
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
	cert.unshifted = points
	cert.shifted = trust_region.unshift(points)

	if not n_points == p:
		raise Exception("currently, have to have all points")

	V = numpy.bmat([
		[basis.evaluate_to_matrix(cert.shifted)],
		[eye(p)]
	])

	for i in range(0, p):
		_test_v(V, basis, cert.shifted)

		# Get maximum value in matrix
		max_value, max_index = _get_max_index(abs(V[i:n_points, i]))

		## Check the poisedness
		if max_index < lagrange_params.xsi and lagrange_params.improve_with_new_points:
			# If still not poised, Then check for new points
			newValue, _ = _maximize_lagrange(bss, V[npoints:h, i], tol, params.getShiftedConstraints())
			maxVal, maxIdx = _replace(cert, i, newValue, npoints, h, V, bss)




		#if max_value < cert.outputXsi and params.improveWithNew:
		#	# If still not poised, Then check for new points
		#	newValue, _ = _maximize_lagrange(bss, V[npoints:h, i], tol, params.getShiftedConstraints())
		#	maxVal, maxIdx = _replace(cert, i, newValue, npoints, h, V, bss)

		if max_value < cert.outputXsi:
			if maxVal < params.minXsi:
				# If still not poised, we are stuck

				for j in range(n_points):
					_, lambdas[j] = _maximize_lagrange(bss, V[n_points:h, j], tol)
				cert.plotIncomplete('images/failed.png', params.radius, maxVal, i, lambdas)
				cert.fail()
				return cert
			print('bumping xsi to ' + str(maxVal))
			cert.outputXsi = maxVal

		# perform pivot
		if not maxIdx == 0:
			otherIdx = maxIdx + i
			_swapRows(V, i, otherIdx)
			_swapRows(cert.shifted, i, otherIdx)
			tmp = cert.indices[i]
			cert.indices[i] = cert.indices[otherIdx]
			cert.indices[otherIdx] = tmp

		# perform LU
		V[:, i] = V[:, i] / V[i, i]
		for j in range(0, p):
			if i == j:
				continue
			V[:, j] = V[:, j] - V[i, j] * V[:, i]

	if params.consOpts is not None and params.consOpts.ellipse is not None:
		cert.unshifted = _unshiftEllipse(cert.shifted, params.consOpts.ellipse)
	else:
		cert.unshifted = _unshift(cert.shifted, params.center, params.radius)
	cert.lmbda = V[npoints:h]
	cert.poised = True

	_testV(V, bss, cert.shifted)
	cert.Lambda = empty(npoints)
	for i in range(npoints):
		_, cert.Lambda[i] = _maximize_lagrange(bss, V[npoints:h, i], tol)
		# if cert.Lambda[i] < 1 - tol and params.improveWithNew:
		# 	print('Found a value of lambda that is less than 1', cert.Lambda[i])
			# raise Exception('Lambda must be greater or equal 1')

	cert.LambdaConstrained = empty(npoints)
	for i in range(npoints):
		_, cert.LambdaConstrained[i] = _maximize_lagrange(bss, V[npoints:h, i], tol, params.getShiftedConstraints())
#		if cert.LambdaConstrained[i] < 1 - tol and params.improveWithNew:
#			print('Found a value of lambda that is less than 1', cert.LambdaConstrained[i])
#			_maximize_lagrange(bss, V[npoints:h, i], tol, params.getShiftedConstraints())
#			raise Exception('Lambda must be greater or equal 1')

	return cert
