
class Bounds:
	def __init__(self):
		self.lbX = None
		self.ubX = None
		self.lbY = None
		self.ubY = None

	def extend(self, x):
		if self.ubX is None or x[0] > self.ubX:
			self.ubX = x[0]
		if self.lbX is None or x[0] < self.lbX:
			self.lbX = x[0]
		if self.lbY is None or x[1] > self.ubY:
			self.ubY = x[1]
		if self.lbY is None or x[1] < self.lbY:
			self.lbY = x[1]

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
		b.lbX = decrease(self.lbX)
		b.ubX = increase(self.ubX)
		b.lbY = decrease(self.lbY)
		b.ubY = increase(self.ubY)
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
