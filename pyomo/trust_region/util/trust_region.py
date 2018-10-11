import abc
import numpy
import matplotlib.pyplot as plt


class TrustRegion(metaclass=abc.ABCMeta):
	def __init__(self):
		pass

	@abc.abstractmethod
	def add_shifted_pyomo_constraints(self, model):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def shift(self, points):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def unshift(self, points):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def shift_row(self, point):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def unshift_row(self, point):
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

	def shift_row(self, point):
		return (point - self.center) / self.radius

	def unshift_row(self, point):
		return point * self.radius + self.center

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

	def add_shifted_pyomo_constraints(self, model):
		model.constraints.add(sum(model.x[i] * model.x[i] for i in model.dimension) <= 1.0)

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
				lambda x: -(x[i] - self.center[i] + self.radius),
				'outer_tr_' + str(i),
				color='r',
				lvls=[-0.1, 0.0]
			)
			plot_object.add_contour(
				lambda x: x[i] - self.center[i] - self.radius,
				'outer_tr_' + str(i),
				color='r',
				lvls=[-0.1, 0.0]
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

	def get_a(self):
		ret_val = numpy.zeros((2 * len(self.center), len(self.center)))
		for i in range(len(self.center)):
			ret_val[2 * i + 0, i] = +1.0
			ret_val[2 * i + 1, i] = -1.0
		return ret_val

	def get_b(self):
		ret_val = numpy.zeros(2 * len(self.center))
		for i in range(len(self.center)):
			ret_val[2 * i + 0] = +self.center[i] + self.radius
			ret_val[2 * i + 1] = -self.center[i] + self.radius
		return ret_val
