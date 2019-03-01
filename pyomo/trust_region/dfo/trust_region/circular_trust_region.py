
import numpy
import matplotlib.pyplot as plt
from trust_region.util.directions import sample_search_directions

from trust_region.dfo.trust_region.trust_region import TrustRegion


class CircularTrustRegion(TrustRegion):
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def to_json(self):
		return {
			'type': 'spherical',
			'center': self.center,
			'radius': self.radius
		}

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

	def add_to_plot(self, plot_object, detailed=True):
		plot_object.add_point(self.center, 'trust region center', color='g', s=20, marker="*")
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

	def sample_shifted_region(self, num_points):
		ret = [numpy.zeros(len(self.center))]
		for d in sample_search_directions(len(self.center), num_points, include_axis=False):
			ret.append(numpy.random.random() * d)
		return ret

	def shift_polyhedron(self, polyhedron):
		A = polyhedron[0]
		b = polyhedron[1]
		return (
			A * self.radius,
			b - numpy.dot(A, self.center)
		)

	def contains(self, point):
		raise numpy.linalg.norm(self.center - point) <= self.radius
