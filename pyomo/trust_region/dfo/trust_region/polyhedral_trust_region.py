
import numpy

from trust_region.dfo.trust_region.trust_region import TrustRegion


class PolyhedralTrustRegion(TrustRegion):
	def __init__(self, l1, polyhedron):
		self.l1 = l1.copy()
		self.polyhedron = polyhedron

	def to_json(self):
		return {
			'ls': self.l1.to_json(),
			'polyhedron': self.polyhedron.to_json()
		}

	def get_polyhedron(self):
		return self.polyhedron.intersect(self.l1.get_polyhedron())

	def contains(self, point):
		return self.l1.contains(point) and self.polyhedron.contains(point)

	def get_shifted_polyhedron(self):
		return self.get_polyhedron().shift(self.l1.center, self.l1.radius)

	def add_shifted_pyomo_constraints(self, model):
		self.get_shifted_polyhedron().add_to_pyomo(model)

	def add_unshifted_pyomo_constraints(self, model):
		self.get_polyhedron().add_to_pyomo(model)

	def shift(self, points):
		return self.l1.shift(points)

	def unshift(self, points):
		return self.l1.unshift(points)

	def shift_row(self, point):
		return self.l1.shift_row(point)

	def unshift_row(self, point):
		return self.l1.unshift_row(point)

	def add_to_plot(self, plot_object, detailed=True):
		# Don't do anything yet
		pass

	def multiply_radius(self, factor):
		return self.l1.multiply_radius(factor)

	def recenter(self, new_center):
		return self.l1.recenter(factor)

	def shift_pyomo_model(self, model):
		raise Exception("Not implemented")

	def sample_shifted_region(self, num_points):
		polyhedron = self.get_shifted_polyhedron()
		ret = [numpy.zeros(len(self.l1.center))]
		while len(ret) + 1 < num_points:
			p = numpy.zeros(len(self.l1.center))
			for i in range(len(self.l1.center)):
				p[i] = 2 * numpy.random.random() - 1
			if polyhedron.contains(p):
				ret.append(p)
		return ret

	def shift_polyhedron(self, polyhedron):
		return self.l1.shift_polyhedron(polyhedron)
