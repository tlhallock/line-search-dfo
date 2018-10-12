
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
			self.ub = x
		if self.lb is None:
			self.lb = x
		for i in range(len(x)):
			if x[i] > self.ub[i]:
				self.ub[i] = x[i]
			if x[i] < self.lb[i]:
				self.lb[i] = x[i]

	def expand(self, factor=1.2):
		def increase(x):
			if x is None:
				return 1.0
			if x > 0.0:
				return factor * x
			else:
				return x / factor

		def decrease(x):
			if x is None:
				return -1.0
			if x < 0.0:
				return factor * x
			else:
				return x / factor

		b = Bounds()
		b.ub = numpy.copy(self.ub)
		b.lb = numpy.copy(self.lb)
		for i in range(len(self.ub)):
			b.ub[i] = increase(b.ub[i])
			b.lb[i] = decrease(b.lb[i])
		return b


class History:
	def __init__(self):
		self.bounds = Bounds()
		self.sample_points = []
		self.objective_values = []

	def add_objective_value(self, x, y):
		self.sample_points.append(x)
		self.objective_values.append(y)
		self.bounds.extend(x)

	def get_plot_bounds(self):
		return self.bounds.expand()
