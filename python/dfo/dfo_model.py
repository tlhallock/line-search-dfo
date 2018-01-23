import matplotlib.pyplot as plt
from numpy import asmatrix
from numpy import asarray
from numpy import dot
from numpy import empty
from numpy import ravel
from numpy.linalg import norm as norm
from numpy.matlib import repmat

from dfo import lagrange
from dfo import polynomial_basis
from utilities import functions
from utilities.boxable_query_set import EvaluationHistory
from utilities.ellipse import getMaximalEllipseContaining
from numpy import concatenate


class MultiFunctionModel:
	def __init__(self, funs, basis, x0, radius=1, initialXsi=1e-1, minXsi=None, consOpts=None):
		self.initialXsi = initialXsi
		self.minXsi = minXsi
		self.functionEvaluations = 0
		self.basis = basis
		self.linearBasis = polynomial_basis.PolynomialBasis(len(x0), 1)
		self.delegates = funs
		self.modelRadius = radius
		self.history = EvaluationHistory(len(x0))
		self.phi = None
		self.cert = None
		self.consOpts = consOpts
		self.currentSet = repmat(x0, basis.basis_dimension, 1)
		self.improve(None)

	def __len__(self):
		return self.size()

	def setNewModelCenter(self, newCenter):
		self.currentSet[0, :] = newCenter
		self.improve()
		# self._improveWithoutNewPoints() ################ Just commented this out!!!!!

	def modelCenter(self):
		""" Returns the model center: namely the first element of the poised set """
		return self.currentSet[0, :]


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
		retVal = empty((self.currentSet.shape[0], len(self.delegates)))
		for i in range(0, self.currentSet.shape[0]):
			otherX, otherY = self.history.get(self.currentSet[i])
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
			return functions.VectorFunction(self.basis.n)
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

	def updateEllipse(self):
		if not self.consOpts.useEllipse:
			return
		ub = self.modelCenter() + self.modelRadius
		lb = self.modelCenter() - self.modelRadius
		aWithRadius = concatenate((
			self.consOpts.A,
			asarray([[1, 0], [-1, 0], [0, 1], [0, -1]])
		))
		bWithRadius = concatenate((
			self.consOpts.b,
			asarray((lb[0], -ub[0], lb[1], -ub[1]))
		))

		self.consOpts.ellipse = getMaximalEllipseContaining(
			aWithRadius,
			bWithRadius,
			self.modelCenter(),
			self.consOpts.tol
		)
		if not self.consOpts.ellipse['success']:
			print('unable to find ellipse')
			getMaximalEllipseContaining(aWithRadius, bWithRadius, self.modelCenter(), self.consOpts.tol)

	def _improveWithoutNewPoints(self):
		""" Determine if the current set is lambda poised, possibly replacing points with points already evaluated """
		self.updateEllipse()
		self.cert = lagrange.computeLagrangePolynomials(
			self.basis,
			self.currentSet,
			lagrange.LagrangeParams(self.modelCenter(), self.modelRadius, False, self.initialXsi, self.consOpts))

		if self.cert.poised:
			self.phi = self.cert.lmbda * self.createY()
			self.currentSet = self.cert.unshifted
			self.cert.unshifted = None

		return self.cert.poised

	def improve(self, plotFile=None):
		""" Ensure that the current set is well poised
			This also evaluates the delegate functions at new points
			This also updates the model based on these new values
		"""
		self.updateEllipse()

		self.cert = lagrange.computeLagrangePolynomials(
			self.basis,
			self.currentSet,
			lagrange.LagrangeParams(self.modelCenter(), self.modelRadius, True, self.initialXsi, self.consOpts))

		if not self.cert.poised:
			return False

		self.currentSet = self.cert.unshifted

		for i in range(self.currentSet.shape[0]):
			self.computeValueFromDelegate(self.currentSet[i, :])

		# update the model
		self.phi = self.cert.lmbda * self.createY()

		if plotFile is not None:
			self.cert.plot(plotFile, self.modelCenter(), self.modelRadius)

		self.cert.unshifted = None

		return self.cert.poised

	def createUnshiftedQuadraticModel(self, other_fun):
		y = asmatrix([other_fun(self.currentSet[i]) for i in range(self.currentSet.shape[0])]).T
		return self.basis.getQuadraticModel(self.cert.lmbda * y)

	def multiplyRadius(self, factor):
		self.modelRadius *= factor

	def interpolate(self, x):
		""" Use the model to predict all function values at the point x """
		return ravel(dot(ravel(self.basis.evaluateRowToRow(
			(x - self.modelCenter()) / self.modelRadius)), self.phi))

	def addPointsToPlot(self, center, rad):
		ax1 = plt.gca()
		# Only plot the points that lie within the plot!
		lie_within_plot = norm(self.cert.unshifted - center, axis=1) < rad
		ax1.scatter(self.currentSet[lie_within_plot, 0], self.currentSet[lie_within_plot, 1], s=10, c='r', marker="x")
		ax1.add_artist(plt.Circle(self.modelCenter(), self.modelRadius, color='g', fill=False))
