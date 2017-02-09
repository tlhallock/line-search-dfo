
from numpy import empty
from numpy.linalg import norm
from dfo import polynomial_basis
from dfo import dfo_model
from numpy import array as arr
import matplotlib.pyplot as plt

from numpy import linspace
from numpy import meshgrid

import dfo
import matplotlib

class Program:
	def __init__(self, name, obj, eq, ineq, x0):
		self.name = name
		self.f = obj
		self.h = eq
		self.g = ineq
		self.x0 = x0
		self.tol = 1e-8
		self.imageNumber = 0

	def getImageNumber(self):
		returnValue = self.imageNumber
		self.imageNumber += 1
		return '%04d' % returnValue

	def hasConstraints(self):
		return self.hasEqualityConstraints() or self.hasInequalityConstraints()


	def hasEqualityConstraints(self):
		return bool(self.h)
	def equalityConstraints(self, x):
		return self.h.evaluate(x)
	def equalityConstraintsJacobian(self, x):
		return self.h.jacobian(x)
	def getNumEqualityConstraints(self):
		if self.h is None:
			return 0
		return self.h.getOutDim()

	def hasInequalityConstraints(self):
		return bool(self.g)
	def inequalityConstraints(self, x):
		return self.g.evaluate(x)
	def inequalityConstraintsJacobian(self, x):
		return self.g.jacobian(x)
	def getNumInequalityConstraints(self):
		if self.g is None:
			return 0
		return self.g.getOutDim()

	def objective(self, x):
		return self.f.evaluate(x)
	def gradient(self, x):
		return self.f.gradient(x)
	def hessian(self, x):
		return self.f.hessian(x)

	def createBasePlotAt(self, centerX):
		matplotlib.rcParams['xtick.direction'] = 'out'
		matplotlib.rcParams['ytick.direction'] = 'out'

		r = self.getRadius()
		x = linspace(centerX[0]-r, centerX[0]+r, num=100)
		y = linspace(centerX[1]-r, centerX[1]+r, num=100)
		X, Y = meshgrid(x, y)

		Z = empty((len(y), len(x)))

		plt.figure()
		plt.title('Current step')

		for i in range(0, len(x)):
			for j in range(0, len(y)):
				Z[j, i] = self.objective(arr([x[i], y[j]]))
		CS = plt.contour(X, Y, Z, 6, colors='k')
		plt.clabel(CS, fontsize=9, inline=1)

		for idx in range(0, self.getNumEqualityConstraints()):
			for i in range(0, len(x)):
				for j in range(0, len(y)):
					Z[j, i] = self.equalityConstraints(arr([x[i], y[j]]))[idx]
			CS = plt.contour(X, Y, Z, 6, colors='r')
			plt.clabel(CS, fontsize=9, inline=1)

		for idx in range(0, self.getNumInequalityConstraints()):
			for i in range(0, len(x)):
				for j in range(0, len(y)):
					Z[j, i] = self.inequalityConstraints(arr([x[i], y[j]]))[idx]
			CS = plt.contour(X, Y, Z, 6, colors='b')
			plt.clabel(CS, fontsize=9, inline=1)

		self.addDetailsToPlot()

		return "images/" + self.name + "_" + self.getImageNumber() + "_program_state.png"

# These methods are to be overriden by a DFO algorithm
	def acceptable(self, newCenter):
		return True
	def converged(self):
		return True
	def addDetailsToPlot(self):
		pass
	def getRadius(self):
		return 10
	def clipToTrustRegion(self, x):
		return x


def _createModel(obj, eq, ineq, x0, b, radius, xsi):

	if b is None:
		b = dfo.polynomial_basis.PolynomialBasis(len(x0), 2)

	equalityIndices = empty(len(eq), dtype=int)
	inequalityIndices = empty(len(eq), dtype=int)

	funs = []
	index = 0

	funs.append(obj)
	objectiveIndex = int(index)
	index += 1

	for i in range(len(eq)):
		funs.append(eq[i])
		equalityIndices[i] = int(index)
		index += 1
	for i in range(len(ineq)):
		funs.append(ineq[i])
		inequalityIndices[i] = int(index)
		index += 1

	model = dfo.dfo_model.MultiFunctionModel(funs, b, x0, radius, xsi)

	return (model, objectiveIndex, equalityIndices, inequalityIndices)


class DfoProgram(Program):
	def __init__(self, name, obj, eq, ineq, x0, b=None, radius=2, xsi=1e-2, plotImprovements=True):
		Program.__init__(self, name, None, None, None, x0)
		self.plotImprovements = plotImprovements

		self.model, self.objectiveIndx, self.equalityIndices, self.inequalityIndices = _createModel(obj, eq, ineq, x0, b, radius, xsi)

		self.radius_increase = 2
		self.radius_decrease = .5
		self.maxDiff = 1

		# :)
		self.reallyGood = 10
		self.goodEnough = .5

		self.rejects = 0

		self._updateModel()

	def acceptable(self, newCenter):
		pva = self.model.testNewModelCenter(newCenter)

		# TODO: create multi dimensional Rho!!!!
		deviation = norm(pva[:,0] - pva[:, 1]) / (pva.shape[0] * max(1, norm(pva[:, 1])))


		if deviation < self.reallyGood:
			if self.model.isNearCenter(newCenter):
				self.model.multiplyRadius(self.radius_decrease)
			else:
				self.model.multiplyRadius(self.radius_increase)

		if deviation < self.goodEnough:
			self.model.setNewModelCenter(newCenter)
			self.model.improve(self.createImprovementFile())
			self._updateModel()
			return True

		if self.model.isLambdaPoised():
			self.model.multiplyRadius(self.radius_decrease)

		self.model.improve(self.createImprovementFile())
		self._updateModel()
		self.rejects += 1
		return False

	def converged(self):
		if self.model.modelRadius < self.tol:
			return True
		self.model.multiplyRadius(self.radius_decrease)
		self.model.improve(self.createImprovementFile())
		self._updateModel()
		return False

	def createImprovementFile(self):
		if self.plotImprovements:
			return 'images/' + self.name + "_" + self.getImageNumber() + '_improve.png'
		else:
			return None

	def _updateModel(self):
		# Currently only using quadratic models!!!!! (regardless of basis...)
		self.f = self.model.getQuadraticModel(self.objectiveIndx)
		self.g = self.model.getQuadraticModels(self.inequalityIndices)
		self.h = self.model.getQuadraticModels(self.equalityIndices)

	def getRadius(self):
		return 2 * self.model.modelRadius

	def hasEqualityConstraints(self):
		return len(self.equalityIndices) > 0
	def hasInequalityConstraints(self):
		return len(self.inequalityIndices) > 0

	def addDetailsToPlot(self):
		ax1 = plt.gca()
		ax1.scatter(self.model.unshifted[:, 0], self.model.unshifted[:, 1], s=10, c='r', marker="x")
		ax1.add_artist(plt.Circle(self.model.modelCenter(), self.model.modelRadius, color='g', fill=False))

	def clipToTrustRegion(self, x):
		if norm(x) > self.model.modelRadius:
			return self.model.modelRadius * x / norm(x)
		else:
			return x

