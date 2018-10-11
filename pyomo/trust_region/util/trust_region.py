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
	def add_to_plot(self, plot_object):
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

	@abc.abstractmethod
	def shift_pyomo_model(self, model):
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

	def add_to_plot(self, plot_object):
		plot_object.ax.add_artist(plt.Circle(self.center, self.radius, color='g', fill=False))

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

	def shift_pyomo_model(self, model):
		class mocked:
			def __init__(self, x):
				self.dimension = model.dimension
				self.x = x
		return mocked([
			(model.x[idx] - self.center[idx]) / self.radius
			for idx in model.dimension
		])


class L1TrustRegion(CircularTrustRegion):
	def add_to_plot(self, plot_object):
		for i in range(len(self.center)):
			plot_object.add_contour(
				lambda x: x[i] - self.center[i] + self.radius,
				'outer_tr_' + str(i),
				color='b',
				lvls=[0.0, -0.1]
			)

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
