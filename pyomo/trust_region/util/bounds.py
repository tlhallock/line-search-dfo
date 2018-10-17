import numpy


class Bounds:
	def __init__(self):
		self.ub = None
		self.lb = None

	def sample(self):
		if self.ub is None or self.lb is None:
			return None
		return self.lb + numpy.multiply(
			numpy.random.random(len(self.ub)), self.ub - self.lb
		)

	def extend(self, x):
		if self.ub is None:
			self.ub = numpy.copy(x)
		if self.lb is None:
			self.lb = numpy.copy(x)

		for i in range(len(x)):
			if x[i] > self.ub[i]:
				self.ub[i] = x[i]
			if x[i] < self.lb[i]:
				self.lb[i] = x[i]

	def expand(self, factor=1.2):
		b = Bounds()
		b.ub = numpy.copy(self.ub)
		b.lb = numpy.copy(self.lb)
		for i in range(len(self.ub)):
			expansion = (factor - 1.0) * (b.ub[i] - b.lb[i])
			b.ub[i] = b.ub[i] + expansion
			b.lb[i] = b.lb[i] - expansion
		return b

	def __str__(self):
		return '[' + str(self.lb) + ' ' + str(self.ub) + ']'
