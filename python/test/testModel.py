

from dfo import polynomial_basis
from dfo import dfo_model
from numpy import array as arr
from numpy import zeros
from numpy.linalg import norm
from utilities import functions
from numpy import random
import unittest



class TestModel(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestModel, self).__init__(*args, **kwargs)

		random.seed(1776)

		n = 2
		degree = 2
		self.b = polynomial_basis.PolynomialBasis(n, degree)
		self.fun1 = functions.Quadratic(arr([[1,0],[0,-2]]), arr([3,2]), 4)
		self.fun2 = functions.Quadratic(zeros((2,2)), arr([3,2]), 4)
		funs = []
		funs.append(self.fun1)
		funs.append(self.fun2)
		radius = 2
		center = arr([5, 5])
		self.set = dfo_model.MultiFunctionModel(funs, self.b, center, radius)
		self.tolerance = .025

	def test_simpleValuesMatch(self):
		x = arr([3, 4])

		y1 = arr([self.fun1.evaluate(x), self.fun2.evaluate(x)])
		y2 = self.set.interpolate(x)

		self.assertTrue(norm(y1 - y2) < self.tolerance)

#		linear = self.set.createLinearModelDepricated(1)
#		self.assertTrue(abs(linear(x)-self.fun2.evaluate(x)) < self.tolerance)

#		qObj = self.set.getQuadraticModel(0)
#		lCon = self.set.getLinearModel(1)

		quadmod = self.set.getQuadraticModels(arr([0, 1], int))
		y2 = quadmod.evaluate(x)

		self.assertTrue(norm(y1 - y2) < self.tolerance)

		newCenter = 10 * (2 * random.random(2) - 1)
		for i in range(50):
			print("testing " + str(i) + " of " + str(50))
			rFactor = .25 + 2 * random.random()
			newRadius = self.set.modelRadius * rFactor
			newCenter = newCenter + self.set.modelRadius / newRadius
			oldlen = self.set.history.size()
			pva = self.set.testNewModelCenter(newCenter)
			newlen = self.set.history.size()

			self.assertTrue(newlen == oldlen + 1)

			deviation = norm(((pva[:, 0] - pva[:, 1]) / pva[:, 1])) / pva.shape[0]
			self.assertTrue(deviation < self.tolerance)
			self.set.setNewModelCenter(newCenter)
			self.set.multiplyRadius(rFactor)
			self.set.improve('images/test_%04d_improve.png' % i)

			quadmod = self.set.getQuadraticModels(arr([0, 1], int))
			for j in range(10):
				x = 10 * (2 * random.random(2) - 1)
				y1 = arr([self.fun1.evaluate(x), self.fun2.evaluate(x)])
				y2 = quadmod.evaluate(x)

				self.assertTrue(norm(y1 - y2) < self.tolerance)







