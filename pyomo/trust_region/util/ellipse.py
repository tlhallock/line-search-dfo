
import numpy

from trust_region.util.trust_region import TrustRegion


class Ellipse(TrustRegion):
	def __init__(self):
		self.center = None
		self.volume = None
		self.ds = None
		self.lambdas = None
		self.scale = 1.0
		self.q = None
		self.q_inverse = None
		self.l = None
		self.l_inverse = None

	def evaluate(self, v, scale=1.0):
		return 1 - 0.5 * numpy.dot(v - self.center, numpy.dot(self.q, v - self.center)) / scale

	def shift_row(self, v):
		return numpy.sqrt(0.5) * numpy.dot(self.l, v - self.center)

	def unshift_row(self, v):
		return self.center + numpy.sqrt(2) * numpy.dot(self.l_inverse, v)

	def shift(self, points):
		return numpy.sqrt(0.5) * numpy.dot(self.l, (points - self.center).T).T

	def unshift(self, points):
		return self.center + numpy.sqrt(2) * numpy.dot(self.l_inverse, points.T).T

	def get_scale_to_include(self, include):
		# 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)) / scale == 0
		# scale - 0.5 * dot(point - xbar, dot(Q, point - xbar)) == 0
		# scale == 0.5 * dot(point - xbar, dot(Q, point - xbar))
		scale = 0.5 * numpy.dot(include - self.center, numpy.dot(self.q, include - self.center))
		return numpy.max(1.0, scale)

	def add_shifted_pyomo_constraints(self, model):
		model.constraints.add(sum(model.x[i] * model.x[i] for i in model.dimension) <= 1.0)

	def add_unshifted_pyomo_constraints(self, model):
		raise Exception("Not implemented")

	def add_to_plot(self, plot_object, detailed=True):
		plot_object.add_contour(
			lambda x: -self.evaluate(x),
			label='ellipse',
			color='g',
			lvls=[-0.1, 0.0]
		)
		if not detailed:
			return
		for d in self.ds:
			plot_object.add_arrow(self.center, self.center + d, color="c")

	def multiply_radius(self, factor):
		raise Exception("Not implemented")

	def recenter(self, new_center):
		raise Exception("Not implemented")

	def shift_pyomo_model(self, model):
		raise Exception("Not implemented")
