from dfo import polynomial_basis

from numpy import inf as infinity
from numpy.linalg import norm as norm
from numpy import zeros
from numpy import bmat as blockmat
from numpy import arange
from numpy import empty
from numpy import eye
from numpy import dot
from numpy import reshape
from numpy import random
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.optimize import minimize
from utilities import sys_utils


class Certification:
	def __init__(self, poisedSet, params):
		self.original = poisedSet
		self.poised = False
		if params.consOpts.ellipse:
			self.shifted = _shiftEllipse(poisedSet, params.consOpts.ellipse)
		else:
			self.shifted = _shift(poisedSet, params.center, params.radius)
		self.lmbda = None
		self.indices = arange(0, poisedSet.shape[0])
		self.unshifted = None
		self.Lambda = None
		self.LambdaConstrained = None
		self.outputXsi = params.initialXsi
		if params.improveWithNew:
			for i in range(1, self.shifted.shape[0]):
				if norm(self.shifted[i, :]) > params.max_radius:
					self.shifted[i, :] = zeros(self.shifted.shape[1])

	def fail(self):
		self.poised = False
		self.shifted = None
		self.lmbda = None
		self.indices = None
		self.unshifted = None
		self.LambdaConstrained = None
		self.outputXsi = None

	def plot(self, filename, center, radius):
		fig = plt.figure()
		fig.set_size_inches(sys_utils.get_plot_size(), sys_utils.get_plot_size())
		ax1 = fig.add_subplot(111)
		ax1.add_artist(plt.Circle(center, radius, color='g', fill=False))
		ax1.scatter(self.original[:, 0], self.original[:, 1], s=10, c='b', marker="+", label='original')
		ax1.scatter(self.unshifted[:, 0], self.unshifted[:, 1], s=10, c='r', marker="x", label='poised')

		# ax1.axis([center[0] - 2 * radius, center[0] + 2 * radius, center[1] - 2 * radius, center[1] + 2 * radius])

		if self.Lambda is not None:
			lambdaStr = "Lambda=" + str(max(self.Lambda))
			ax1.text(center[0], center[1], lambdaStr)

		plt.legend(loc='lower left')
		fig.savefig(filename)
		plt.close()

	def plotIncomplete(self, filename, radius, newXsi, iteration, lambdas):
		fig = plt.figure()
		fig.set_size_inches(sys_utils.get_plot_size(), sys_utils.get_plot_size())
		ax1 = fig.add_subplot(111)
		center = self.shifted[0, :]

		ax1.add_artist(plt.Circle(center, radius, color='g', fill=False))
		ax1.scatter(self.shifted[:, 0], self.shifted[:, 1], s=10, c='r', marker="x", label='failed')

		# ax1.axis([center[0] - 2 * radius, center[0] + 2 * radius, center[1] - 2 * radius, center[1] + 2 * radius])

		lambdaStr = "xsi=" + str(newXsi) + " on iteration " + str(iteration) + " Lambda=" + str(max(lambdas))
		ax1.text(center[0], center[1], lambdaStr)

		plt.legend(loc='lower left')
		fig.savefig(filename)
		plt.close()

class LagrangeParams:
	def __init__(self, center, radius, improve, initialXsi, consOpts=None, minXsi=1e-10):
		self.improveWithNew = improve
		self.initialXsi = initialXsi
		self.minXsi = minXsi
		self.radius = radius
		self.center = center
		self.max_radius = 1.1
		self.consOpts = consOpts

	def getShiftedConstraints(self):

		# I have spent a while staring at this code
		# I can't figure out why it doesn't work.
		# There is something I don't understand about python.
		# I also tried the map function.

		# constraints = [{
		# 	'type': constraint['type'],
		# 	'fun':  lambda x: constraint['fun'](x * self.radius + self.center),
		# 	'jac':  lambda x: constraint['jac'](x * self.radius + self.center) * self.radius
		# } for constraint in self.consOpts.constraints]

		constraints = [{
			'type': 'ineq',
			'fun': lambda x: self.consOpts.constraints[0]['fun'](x * self.radius + self.center),
			'jac': lambda x: self.consOpts.constraints[0]['jac'](x * self.radius + self.center) * self.radius
		}, {
			'type': 'ineq',
			'fun': lambda x: self.consOpts.constraints[1]['fun'](x * self.radius + self.center),
			'jac': lambda x: self.consOpts.constraints[1]['jac'](x * self.radius + self.center) * self.radius
		}]

		return constraints

def _shiftEllipse(set, ellipse):
	retVal = empty(set.shape)
	for i in range(0, set.shape[0]):
		retVal[i, :] = ellipse['shift'](set[i, :])
	return retVal

def _unshiftEllipse(set, ellipse):
	retVal = empty(set.shape)
	for i in range(0, set.shape[0]):
		retVal[i, :] = ellipse['unshift'](set[i, :])
	return retVal

def _shift(set, center, radius):
	retVal = empty(set.shape)
	for i in range(0, set.shape[0]):
		retVal[i, :] = (set[i,:] - center) / radius
	return retVal

def _unshift(set, center, radius):
	retVal = empty(set.shape)
	for i in range(0, set.shape[0]):
		retVal[i, :] = set[i,:] * radius + center
	return retVal

def _testV(V, basis, poisedSet):
	p = basis.basis_dimension
	npoints = poisedSet.shape[0]
	h = npoints + p
	if norm(V[0:npoints, :] - basis.evaluateMatToMat(poisedSet) * V[npoints:h, :]) > 1e-3:
		raise Exception("did not work")

def _getMaxIdx(max):
	idx = 0
	val = max[0,0]
	for i in range(0, max.shape[0]):
		newVal = max[i, 0]
		if newVal > val:
			idx = i
			val = newVal
	return (val, idx)

def _swapRows(mat, idx1, idx2):
	mat[[idx1, idx2], :] = mat[[idx2, idx1], :]

def _maximize_lagrange_quad(basis, row, tol, constraints):
	""" This method uses the fact that we are modelling with quadratics..."""
	quadmodel = basis.getQuadraticModel(row)

	cons = [{'type': 'ineq',
			 'fun': lambda x: 1 - dot(x, x),
			 'jac': lambda x: reshape(-2 * x, (1, basis.n))},
	]

	if constraints is not None:
		cons += constraints

	# Here, I should solve a program to find a feasible point, instead of trying several different points.

	feasibleStart = minimize(lambda x: sum([-v if v < 0 else 0 for v in [c['fun'](x) for c in cons]]),
	 x0 = 2 * random.rand(basis.n) - 1, method='Nelder-Mead', options={"disp": False, "maxiter": 1000}, tol=tol)

	minimumResult = None
	if feasibleStart.success:
		minimumResult = minimize(quadmodel.evaluate, jac=quadmodel.gradient, x0=feasibleStart.x,
				constraints=cons, method='SLSQP', options={"disp": False, "maxiter": 1000},
				tol=tol)
	else:
		print('not able to find a feasible starting point.')
	count = 0
	while count < 1000 and (minimumResult is None or not minimumResult.success or norm(minimumResult.x) >= 1 + 1e-4 + tol):
		minimumResult = minimize(quadmodel.evaluate, jac=quadmodel.gradient, x0=random.random(basis.n) / 10,
							constraints=cons, method='SLSQP', options={"disp": False, "maxiter": 1000}, tol=tol)
		count += 1
	if not minimumResult.success or norm(minimumResult.x) >= 1 + 1e-4:
		# Running this one last time with display = true
		minimumResult = minimize(quadmodel.evaluate, jac=quadmodel.gradient, x0=random.random(basis.n) / 10,
								 constraints=cons, method='SLSQP', options={"disp": True, "maxiter": 1000}, tol=tol)
		print(minimumResult)
		raise Exception('Unable to solve the trust region sub problem')


	count = 0
	maximumResult = None
	if feasibleStart.success:
		maximumResult = minimize(lambda x: -quadmodel.evaluate(x), jac=lambda x: -quadmodel.gradient(x), x0=feasibleStart.x,
						constraints=cons, method='SLSQP', options={"disp": False, "maxiter": 1000}, tol=tol)
	while count < 1000 and (maximumResult is None or not maximumResult.success or norm(maximumResult.x) >= 1 + 1e-4 + tol):
		maximumResult = minimize(lambda x: -quadmodel.evaluate(x), jac=lambda x: -quadmodel.gradient(x), x0=random.random(basis.n) / 10,
						constraints=cons, method='SLSQP', options={"disp": False, "maxiter": 1000}, tol=tol)
		count += 1
	if not maximumResult.success or norm(maximumResult.x) >= 1 + 1e-4:
		raise Exception('Unable to solve the trust region sub problem')

	if abs(minimumResult.fun) > abs(maximumResult.fun):
		return minimumResult.x, abs(minimumResult.fun)
	else:
		return maximumResult.x, abs(maximumResult.fun)

def _maximize_lagrange(basis, row, tol, constraints=None):
	return _maximize_lagrange_quad(basis, row, tol, constraints)

def _replace(cert, i, newValue, npoints, h, V, b):
	cert.shifted[i] = newValue
	V[i] = dot(b.evaluateRowToRow(newValue), V[npoints:h, :])
	cert.indices[i] = -1
	_testV(V, b, cert.shifted)
	return _getMaxIdx(abs(V[i:npoints, i]))


def computeLagrangePolynomials(bss, poisedSet, params, history=None, tol=1e-8):
	p = bss.basis_dimension
	npoints = poisedSet.shape[0]
	h = npoints + p

	cert = Certification(poisedSet, params)

	if not npoints == p:
		raise Exception("currently, have to have all points")

	V = blockmat([[bss.evaluateMatToMat(cert.shifted)], [eye(p)]])

	for i in range(0, p):
		_testV(V, bss, cert.shifted)

		# Get maximum value in matrix
		maxVal, maxIdx = _getMaxIdx(abs(V[i:npoints, i]))

		# Check the poisedness
		if maxVal < cert.outputXsi and params.improveWithNew:
			# If still not poised, Then check for new points
			newValue, _ = _maximize_lagrange(bss, V[npoints:h, i], tol, params.getShiftedConstraints())
			maxVal, maxIdx = _replace(cert, i, newValue, npoints, h, V, bss)

		if maxVal < cert.outputXsi:
			if maxVal < params.minXsi:
				# If still not poised, we are stuck
				lambdas = empty(npoints)
				for j in range(npoints):
					_, lambdas[j] = _maximize_lagrange(bss, V[npoints:h, j], tol)
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

	if params.consOpts.ellipse is not None:
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

def computeRegressionPolynomials(basis, shifted):
	p = basis.basis_dimension
	npoints = shifted.shape[0]
	a = basis.evaluateMatToMat(shifted)
	c,_,_,_ = lstsq(a, eye(npoints))
	return c