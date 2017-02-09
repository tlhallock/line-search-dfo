from dfo import polynomial_basis

from numpy import inf as infinity
from numpy.linalg import norm as norm
from numpy import zeros
from numpy import bmat as blockmat
from numpy import array as arr
from numpy import arange
from numpy import empty
from numpy import eye
from numpy import dot
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

from scipy.optimize import minimize



class Certification:
	def __init__(self, poisedSet, params):
		self.original = poisedSet
		self.poised = False
		self.shifted = _shift(poisedSet, params.center, params.radius)
		self.lmbda = None
		self.indices = arange(0, poisedSet.shape[0])
		self.unshifted = None
		self.Lambda = None

		if params.onlyInTrustRegion and params.improveWithNew:
			for i in range(1, self.shifted.shape[0]):
				if norm(self.shifted[i, :]) > params.radius:
					self.shifted[i, :] = zeros(self.shifted.shape[1])


	def fail(self):
		self.poised = False
		self.shifted = None
		self.lmbda = None
		self.indices = None
		self.unshifted = None

	def plot(self, filename, center, radius):
		fig = plt.figure()
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

class LagrangeParams:
	def __init__(self, center, radius, improve, xsi):
		self.improveWithNew = improve
		self.xsi = xsi
		self.radius = radius
		self.center = center
		self.maxL = 2
		self.onlyInTrustRegion = False




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
	h = npoints + p;
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

def _getBestExistingPoint(basis, row, history):
	""" Look through points in history and try to replace the current row """
	#  Priority TODO: implement this!!!!!

	found = False
	maxX = None
	# maxY = -1
	# for point in history's keys:
	#	if too far away:
	#		continue
	#	y = dot(basis.evaluateRowToRow(point), row)
	#	if y > abs(maxY)
	#		y = abs(maxY)
	#		maxX = point
	#		found = True

	return (found, maxX)

def _maximize_lagrange(basis, row):
	""" Need to evaluate jacobian and hessians of basis... (won't be that hard)
	 This should allow a better method than nelder mead"""

	def thePoly(x):
		return dot(basis.evaluateRowToRow(x), row)
	#		def constraint(x):
	#			return radius - norm(x - center)

	minimumResult = minimize(lambda x: thePoly(x) if norm(x) < 1 else infinity,
							 zeros(basis.n), method='Nelder-Mead',
							 #				constraints = {'type':'ineq', 'fun': constraint},
							 options={'xtol': 1e-8, 'disp': False})

	maximumResult = minimize(lambda x: -thePoly(x) if norm(x) < 1 else infinity,
							 zeros(basis.n), method='Nelder-Mead',
							 #				constraints = {'type':'ineq', 'fun': constraint},
							 options={'xtol': 1e-8, 'disp': False})
	if abs(maximumResult.fun) > 1e300 or abs(minimumResult.fun) > 1e300:
		raise Exception('Too big!!!')

	if abs(minimumResult.fun) > abs(maximumResult.fun):
		return minimumResult.x, abs(minimumResult.fun)
	else:
		return maximumResult.x, abs(maximumResult.fun)


def _replace(cert, i, newValue, npoints, h, V, b):
	cert.shifted[i] = newValue
	V[i] = b.evaluateRowToRow(newValue) * V[npoints:h, :]
	cert.indices[i] = -1
	_testV(V, b, cert.shifted)
	return _getMaxIdx(abs(V[i:npoints, i]))


def computeLagrangePolynomials(bss, poisedSet, params, history=None):
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
		if maxVal < params.xsi or maxVal > params.maxL:
			# First, check for an existing point to replace.
			found, newValue = _getBestExistingPoint(bss, V[npoints:h, i], history)
			if found:
				maxVal, maxIdx = _replace(cert, i, newValue, npoints, h, V, bss)

		if (maxVal < params.xsi or maxVal > params.maxL) and params.improveWithNew:
			# If still not poised, Then check for new points
			newValue, _ = _maximize_lagrange(bss, V[npoints:h, i])
			maxVal, maxIdx = _replace(cert, i, newValue, npoints, h, V, bss)

		if maxVal < params.xsi:
			# If still not poised, we are stuck
			cert.fail()
			return cert

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

	cert.unshifted = _unshift(cert.shifted, params.center, params.radius)
	cert.lmbda = V[npoints:h]
	cert.poised = True

	cert.Lambda = empty(npoints)
	for i in range(npoints):
		_, cert.Lambda[i] = _maximize_lagrange(bss, V[npoints:h, i])

	return cert



def computeRegressionPolynomials(basis, shifted):
	p = basis.basis_dimension
	npoints = shifted.shape[0]
	a = basis.evaluateMatToMat(shifted)
	c,_,_,_ = lstsq(a, eye(npoints))
	return c