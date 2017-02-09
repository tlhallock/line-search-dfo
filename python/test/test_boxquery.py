import unittest

from numpy import array as arr
from numpy import linspace


from utilities.boxable_query_set import EvaluationHistory


class TestNonDomSet(unittest.TestCase):

	def testProperQuery(self):
		history = EvaluationHistory(2)

		N = 10
		M = 10

		for i in linspace(-1, 1, N):
			for j in linspace(-1, 1, M):
				history.add(arr([i, j]), i*j)

		self.assertTrue(N*M == history.size())

		count = 0
		for x, y in history.getBox(arr([0, 0]), .2):
			count += 1
		self.assertTrue(count == 4)

		count = 0
		for x, y in history.getBox(arr([0, 0]), .8):
			count += 1

		self.assertTrue(count == 64)



