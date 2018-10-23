
import numpy

from trust_region.dfo.trust_region.trust_region import TrustRegion
from trust_region.util.directions import sample_search_directions
from trust_region.dfo.trust_region.ellipse import _shift_polyhedron


class ScaledEllipse(TrustRegion):
	def __init__(self, ellipse, scale, A, b):
		self.ellipse = ellipse
		self.scale = scale
		self.A = A
		self.b = b

	def evaluate(self, v, scale=1.0):
		return 1 - 0.5 * numpy.dot(v - self.ellipse.center, numpy.dot(self.ellipse.q, v - self.ellipse.center)) / self.scale

	def shift_row(self, v):
		return numpy.sqrt(0.5) * numpy.dot(self.ellipse.l, v - self.ellipse.center) / numpy.sqrt(self.scale)

	def unshift_row(self, v):
		return self.ellipse.center + numpy.sqrt(2) * numpy.dot(self.ellipse.l_inverse, v) * numpy.sqrt(self.scale)

	def shift(self, points):
		return numpy.sqrt(0.5) * numpy.dot(self.ellipse.l, (points - self.ellipse.center).T).T / numpy.sqrt(self.scale)

	def unshift(self, points):
		return self.ellipse.center + numpy.sqrt(2) * numpy.dot(self.ellipse.l_inverse, points.T).T * numpy.sqrt(self.scale)

	def get_shifted_polyhedron(self):
		return _shift_polyhedron(self.A, self.b, self.scale, self.ellipse.l_inverse, self.ellipse.center)

	def add_shifted_pyomo_constraints(self, model):
		model.constraints.add(sum(model.x[i] * model.x[i] for i in model.dimension) <= 1.0)
		shifted_A, shifted_b = self.get_shifted_polyhedron()
		for idx in range(shifted_A.shape[0]):
			model.constraints.add(sum(model.x[i] * shifted_A[idx, i] for i in range(shifted_A.shape[1])) <= shifted_b[idx])

	def add_unshifted_pyomo_constraints(self, model):
		raise Exception("Not implemented")

	def add_to_plot(self, plot_object, detailed=True, color='g'):
		# We should just call ellipse.add_to_plot
		plot_object.add_point(self.ellipse.center, 'trust region center', color=color, s=20, marker="*")
		plot_object.add_contour(
			lambda x: -self.evaluate(x),
			label='scaled_ellipse',
			color=color,
			lvls=[-0.1, 0.0]
		)
		plot_object.add_contour(
			lambda v: 1 - 0.5 * numpy.dot(v - self.ellipse.center, numpy.dot(self.ellipse.q, v - self.ellipse.center)),
			label='unscaled_ellipse',
			color='r',
			lvls=[0.0]
		)
		if not detailed or self.ellipse.ds is None:
			return
		for d in self.ellipse.ds:
			plot_object.add_arrow(self.ellipse.center, self.ellipse.center + d, color="c")

	def multiply_radius(self, factor):
		raise Exception("Not implemented")

	def recenter(self, new_center):
		raise Exception("Not implemented")

	def shift_pyomo_model(self, model):
		raise Exception("Not implemented")

	def sample_shifted_region(self, num_points):
		A, b = self.get_shifted_polyhedron()
		ret = [numpy.zeros(len(self.ellipse.center))]
		while len(ret) + 1 < num_points:
			p = 2 * numpy.random.random(len(self.ellipse.center)) - 1
			p *= numpy.random.random() / numpy.linalg.norm(p)
			if (numpy.dot(A, p) > b).any():
				continue
			ret.append(p)
		return ret

	def shift_polyhedron(self, polyhedron):
		return self.ellipse.shift_polyhedron(polyhedron)

	def contains(self, point):
		return self.evaluate(point) >= 0.0 and (numpy.dot(A, point) <= b).all()

