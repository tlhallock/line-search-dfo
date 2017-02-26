

from dfo import polynomial_basis
from dfo import dfo_model
from numpy import array as arr
from numpy import zeros
from numpy.linalg import norm
from utilities import functions
from numpy import random
import unittest

import numpy
from numpy import linspace
from numpy import ones


class TestModel(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestModel, self).__init__(*args, **kwargs)
		self.tolerance = .025
		n = 2
		degree = 2
		self.b = polynomial_basis.PolynomialBasis(n, degree)
		self.radius = 2
		self.center = arr([5, 5])
		dnum = 100
		drad = .75
		self.elements = 2 ** linspace(-drad, drad, dnum)
		self.probabilites = ones(dnum) / dnum

		random.seed(1776)

	def getRFactor(self):
		return random.choice(self.elements, 1, p=self.probabilites)[0]

	def test_simpleValuesMatch(self):
		fun1 = functions.Quadratic(arr([[1,0],[0,-2]]), arr([3,2]), 4)
		fun2 = functions.Quadratic(zeros((2,2)), arr([3,2]), 4)
		funs = [fun1, fun2]
		set = dfo_model.MultiFunctionModel(funs, self.b, self.center, self.radius)
		x = arr([3, 4])

		y1 = arr([fun1.evaluate(x), fun2.evaluate(x)])
		y2 = set.interpolate(x)

		self.assertTrue(norm(y1 - y2) < self.tolerance)

		quadmod = set.getQuadraticModels(arr([0, 1], int))
		y2 = quadmod.evaluate(x)

		self.assertTrue(norm(y1 - y2) < self.tolerance)

		newCenter = 10 * (2 * random.random(2) - 1)
		for i in range(50):
			print("testing " + str(i) + " of " + str(50))
			rFactor = self.getRFactor()
			newRadius = set.modelRadius * rFactor
			newCenter = newCenter + set.modelRadius / newRadius
			oldlen = set.history.size()
			pva = set.testNewModelCenter(newCenter)
			newlen = set.history.size()

			self.assertTrue(newlen == oldlen + 1)

			deviation = norm(((pva[:, 0] - pva[:, 1]) / pva[:, 1])) / pva.shape[0]
			self.assertTrue(deviation < self.tolerance)
			set.setNewModelCenter(newCenter)
			set.multiplyRadius(rFactor)
			set.improve('images/test_%04d_improve.png' % i)

			quadmod = set.getQuadraticModels(arr([0, 1], int))
			for j in range(10):
				x = 10 * (2 * random.random(2) - 1)
				y1 = arr([fun1.evaluate(x), fun2.evaluate(x)])
				y2 = quadmod.evaluate(x)

				self.assertTrue(norm(y1 - y2) < self.tolerance)

	def test_bothModels(self):
		fun1 = functions.DistanceToCircle(arr([ 10,  10]), .5)
		fun2 = functions.DistanceToCircle(arr([-10, -10]), 5)
		set = dfo_model.MultiFunctionModel([fun1, fun2], self.b, self.center, self.radius)
		set.improve(None)
		center = arr([3,4])

		for i in range(50):
			print("testing " + str(i) + " of " + str(50))
			rFactor = self.getRFactor()
			newRadius = set.modelRadius * rFactor
			center = center + set.modelRadius / newRadius
			set.testNewModelCenter(center)
			set.setNewModelCenter(center)
			set.multiplyRadius(rFactor)
			set.improve('images/test_both_%04d_improve.png' % i)

			quadmod1 = set.getQuadraticModels(arr([0, 1], int))
			quadmod2 = set.getQuadraticModels2(arr([0, 1], int))
			for j in range(10):
				x = center + 10 * (2 * random.random(2) - 1)
				y1 = quadmod1.evaluate(x)
				y2 = quadmod2.evaluate(x)

				self.assertTrue(norm(y1 - y2) < self.tolerance)

				y1 = quadmod1.jacobian(x)
				y2 = quadmod2.jacobian(x)

				self.assertTrue(norm(y1 - y2) < self.tolerance)







# This doesn't work, but I wanted to make a distribution
# with expected value of 1 after multiplying the results of sampling it...
#
# def f(x):
# 	if x < 1:
# 		return .5 * x**3
# 	else:
# 		return 1 - 1 / (2 * x**3)
#
#
# def sample():
# 	x = random.random()
# 	if x < .5:
# 		return (2*x)**(1/3)
# 	else:
# 		return 1/((2*(1-x))**(1/3))
#
#
# def sample():
# 	x = random.random()
# 	if x < .5:
# 		return 1/2
# 	else:
# 		return 2
#
#
# def multiply(n=100):
# 	c = 1
# 	for i in range(n):
# 		c *= sample()
# 		print(c)
# 	return c
#
# def average(m, n):
# 	sum = 0
# 	for i in range(m):
# 		sum += multiply(n)
# 	return sum / m
#
#
# average(1000,1000)




