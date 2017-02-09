import unittest

import numpy

from utilities.nondom import NonDomSet


class TestNonDomSet(unittest.TestCase):
	def test_constructor(self):
		nondom = NonDomSet()

	def test_remove_on_add(self):
		nondom = NonDomSet()
		nondom.add(numpy.array([1, 2, 3]))
		nondom.add(numpy.array([3, 1, 2]))
		nondom.add(numpy.array([2, 3, 1]))
		nondom.add(numpy.array([-1, 5, 5]))
		self.assertTrue(nondom.size() == 4)
		nondom.add(numpy.array([0, 0, 0]))
		self.assertTrue(nondom.size() == 2)
		nondom.add(numpy.array([-1, -1, -1]))
		self.assertTrue(nondom.size() == 1)


