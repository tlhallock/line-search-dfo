
import numpy

from trust_region.util.bounds import Bounds


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

	def add_to_plot(self, plot_object):
		plot_object.add_points(
			numpy.array(self.sample_points),
			label='history',
			color='k',
			s=20,
			marker="x"
		)
