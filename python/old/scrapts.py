
# These methods are to be overriden by a DFO algorithm
	def acceptable(self, newCenter):
		return True
	def converged(self):
		return True
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

	def _improve(self):
		self.model.improve(self.createImprovementFile())
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
			self._improve()
			return True

		if self.model.isLambdaPoised():
			self.model.multiplyRadius(self.radius_decrease)

		self._improve()
		self.rejects += 1
		return False

	def converged(self):
		if self.model.modelRadius < self.tol:
			return True
		self.model.multiplyRadius(self.radius_decrease)
		self._improve()
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

	def clipToTrustRegion(self, x):
		if norm(x) > self.model.modelRadius:
			return self.model.modelRadius * x / norm(x)
		else:
			return x
