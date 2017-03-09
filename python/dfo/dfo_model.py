from dfo import polynomial_basis
from dfo import dfo_model

from numpy.linalg import norm as norm
from numpy import empty
from numpy.matlib import repmat
from numpy import dot
from numpy import asmatrix
from numpy import ravel
from utilities.boxable_query_set import EvaluationHistory
import matplotlib.pyplot as plt

from utilities import functions


from scipy.optimize import minimize



from dfo import polynomial_basis
from dfo import lagrange


class MultiFunctionModel:
	def __init__(self, funs, basis, x0, radius=1, xsi=1e-2):
		self.xsi = xsi
		self.functionEvaluations = 0
		self.basis = basis
		self.linearBasis = polynomial_basis.PolynomialBasis(len(x0), 1)
		self.delegates = funs
		self.modelRadius = radius
		self.history = EvaluationHistory(len(x0))
		self.phi = None
		self.lmbda = None

		self.shifted = None
		self.unshifted = repmat(x0, basis.basis_dimension, 1)
		self.improve(None)

	def __len__(self):
		return self.size()

	def setNewModelCenter(self, newCenter):
		self.unshifted[0, :] = newCenter
		self._improveWithoutNewPoints()

	def modelCenter(self):
		""" Returns the model center: namely the first element of the poised set """
		return self.unshifted[0, :]


	def computeValueFromDelegate(self, x):
		""" computes the function values at x by calling the delegate methods
		returns a tuple:
			the first element is the computed value
			the second element is true if the delegate was actually called
		"""
		newX, newY = self.history.get(x)
		if newY is not None:
			return newY, False

		self.functionEvaluations += 1
		y = empty(len(self.delegates))
		for j in range(0, len(self.delegates)):
			y[j] = self.delegates[j].evaluate(x)

		self.history.add(x, y)

		return y, True


	def createY(self):
		""" Creates the matrix of function values of the current poised set

		There are as many rows as the number of points in the set
		There are as many columns as the number of functions being modeled
		"""
		retVal = empty((self.unshifted.shape[0], len(self.delegates)))
		for i in range(0, self.unshifted.shape[0]):
			otherX, otherY = self.history.get(self.unshifted[i])
			if otherY is None:
				raise Exception("This is a problem...")
			retVal[i] = otherY
		return retVal

	def getQuadraticModel(self, var):
		""" Create a quadratic function modelling the function at index var """
		return self.basis.getQuadraticModel(ravel(self.phi[:, var])).shift(-self.modelCenter(), self.modelRadius)

	def getQuadraticModels2(self, indices):
		fun = functions.VectorFunction(self.basis.n)
		for i in indices:
			fun.add(self.getQuadraticModel(i))
		return fun

	def getQuadraticModels(self, indices):
		if len(indices) == 0:
			return functions.EmptyFunction()
		fun = functions.VectorValuedQuadratic(self.basis.n, len(indices))
		idx=0
		for i in indices:
			fun.set(idx, self.getQuadraticModel(i))
			idx += 1
		return fun

	def getLinearModel(self, var):
		""" Creates a linear function modelling the function at index var """
		return self.basis.getLinearModel(ravel(self.phi[:, var])).shift(-self.modelCenter(), self.modelRadius)

	def isLambdaPoised(self):
		return self._improveWithoutNewPoints()

	def _setUnshiftedFromCert(self, cert):
		self.shifted = cert.shifted
		self.unshifted = cert.unshifted

		# the following would attempt to keep the old points...
		# newunshifted = empty(self.unshifted.shape)
		# for i in range(self.unshifted.shape[0]):
		# 	if cert.indices[i] > 0:
		# 		newunshifted[i, :] = self.unshifted[cert.indices[i], :]
		# 	else:
		# 		newunshifted[i, :] = cert.unshifted[i, :]
		# self.unshifted = newunshifted


	def _improveWithoutNewPoints(self):
		""" Determine if the current set is lambda poised, possibly replacing points with points already evaluated """
		cert = lagrange.computeLagrangePolynomials(
			self.basis,
			self.unshifted,
			lagrange.LagrangeParams(self.modelCenter(), self.modelRadius, False, self.xsi))

		if cert.poised:
			self._setUnshiftedFromCert(cert)
			self.lmbda = cert.lmbda
			self.phi = cert.lmbda * self.createY()

		return cert.poised

	def improve(self, plotFile):
		""" Ensure that the current set is well poised
			This also evaluates the delegate functions at new points
			This also updates the model based on these new values
		"""
		cert = lagrange.computeLagrangePolynomials(
			self.basis,
			self.unshifted,
			lagrange.LagrangeParams(self.modelCenter(), self.modelRadius, True, self.xsi))

		self._setUnshiftedFromCert(cert)

		for i in range(self.unshifted.shape[0]):
			self.computeValueFromDelegate(self.unshifted[i, :])

		# update the model
		self.lmbda = cert.lmbda
		self.phi = cert.lmbda * self.createY()

		if plotFile is not None:
			cert.plot(plotFile, self.modelCenter(), self.modelRadius)

		return cert.poised

	def createUnshiftedQuadraticModel(self, other_fun):
		y = asmatrix([other_fun(self.unshifted[i]) for i in range(self.unshifted.shape[0])]).T
		return self.basis.getQuadraticModel(self.lmbda * y)

	def multiplyRadius(self, factor):
		self.modelRadius *= factor

	def interpolate(self, x):
		""" Use the model to predict all function values at the point x """
		return ravel(dot(ravel(self.basis.evaluateRowToRow(
			(x - self.modelCenter()) / self.modelRadius)), self.phi))

	def addPointsToPlot(self, center, rad):
		ax1 = plt.gca()
		# Only plot the points that lie within the plot!
		lie_within_plot = norm(self.unshifted - center, axis=1) < rad
		ax1.scatter(self.unshifted[lie_within_plot, 0], self.unshifted[lie_within_plot, 1], s=10, c='r', marker="x")
		ax1.add_artist(plt.Circle(self.modelCenter(), self.modelRadius, color='g', fill=False))

		# def isNearCenter(self, newValue):
		# 	return norm(newValue - self.unshifted[0, :]) < self.modelRadius / 2

#	def createLinearModelDepricated(self, var):
#		""" Old code """
#		yvals = self.createYByIndex(var)
#		proj = lagrange.computeRegressionPolynomials(self.linearBasis, self.cert.shifted)
#		c = self.modelCenter()
#		r = self.modelRadius
#		def innerFunc(x):
#			return dot(yvals, dot(self.linearBasis.evaluateRowToRow((x - c) / r), proj))
#		return innerFunc
#
#	def interpolateByIndex(self, x, var):
#		""" Same as interpolate, but only interpolates function at index var """
#		dot(self.createYByIndex(var), ravel(self.basis.evaluateRowToRow(
#			(x - self.modelCenter()) / self.modelRadius) * self.cert.lmbda))
#
#	def createYByIndex(self, var):
#		""" Creates the vector of function values of the current poised set
#
#		This only creates the vector for the function at index var
#
#		"""
#		retVal = empty(self.unshifted.shape[0])
#		for i in range(0, self.unshifted.shape[0]):
#			y = self.history[tuple(self.unshifted[i])]
#			retVal[i] = y[var]
#		return retVal


	# def testNewModelCenter(self, x):
	# 	""" Returns a matrix of predicted versus actual function values
	#
	# 	There are as many rows as functions
	# 	There are two columns:
	# 		one column for the predicted values
	# 		one column for the actual values
	# 	"""
	# 	predictedVersusActual = empty((len(self.delegates), 2))
	# 	predictedVersusActual[:, 0] = self.interpolate(x)
	#
	# 	actual, evaluated = self.computeValueFromDelegate(x)
	#
	# 	predictedVersusActual[:, 1] = actual
	#
	# 	return predictedVersusActual


