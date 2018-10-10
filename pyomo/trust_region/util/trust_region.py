import abc
import numpy
import matplotlib.pyplot as plt


class TrustRegion(metaclass=abc.ABCMeta):
	def __init__(self):
		pass

	def add_shifted_pyomo_constraints(self, model):
		return sum(model.x[i] * model.x[i] for i in model.dimension) <= 1.0

	@abc.abstractmethod
	def shift(self, points):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def unshift(self, points):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def add_to_plot(self, ax):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def multiply_radius(self, factor):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def recenter(self, new_center):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def add_unshifted_pyomo_constraints(self, model):
		raise Exception("Not implemented")


class CircularTrustRegion(TrustRegion):
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def shift_vector(self, vector):
		return (vector - self.center) / self.radius

	def shift(self, points):
		shifted = numpy.empty(points.shape)
		for i in range(0, points.shape[0]):
			shifted[i, :] = (points[i, :] - self.center) / self.radius
		return shifted

	def unshift(self, points):
		unshifted = numpy.empty(points.shape)
		for i in range(0, points.shape[0]):
			unshifted[i, :] = points[i, :] * self.radius + self.center
		return unshifted

	def add_to_plot(self, ax):
		ax.add_artist(plt.Circle(self.center, self.radius, color='g', fill=False))

	def multiply_radius(self, factor):
		self.radius *= factor

	def recenter(self, new_center):
		self.center = new_center

	def add_unshifted_pyomo_constraints(self, model):
		model.constraints.add(
			sum(
				(model.x[i] - self.center[i]) * (model.x[i] - self.center[i])
				for i in model.dimension
			) <= self.radius * self.radius
		)


class L1TrustRegion(CircularTrustRegion):
	def add_to_plot(self, ax):
		ax.add_artist(plt.Circle(self.center, self.radius, color='r', fill=False))
		raise Exception("not implemented")

	def add_shifted_pyomo_constraints(self, model):
		raise Exception("not implemented yet")

	def add_unshifted_pyomo_constraints(self, model):
		for idx in range(len(self.center)):
			model.constraints.add(
				model.x[idx] <= self.center[idx] + self.radius
			)
			model.constraints.add(
				model.x[idx] >= self.center[idx] - self.radius
			)
