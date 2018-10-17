
import numpy
import matplotlib.pyplot as plt


from trust_region.dfo.trust_region.trust_region import TrustRegion

def _add_constraints_to_pyomo(model, A, b):
	for r in range(A.shape[0]):
		model.constraints.add(
			sum(model.x[c] * A[r, c] for c in model.dimension) <= b[r]
		)


class PolyhedralTrustRegion(TrustRegion):
	def __init__(self, l1, constraints_a, constraints_b):
		self.l1 = l1.copy()
		self.constraints_a = constraints_a
		self.constraints_b = constraints_b

	def get_polyhedron(self):
		return numpy.array(numpy.bmat([
			[self.l1.get_a()],
			[self.constraints_a]
		])), numpy.array(numpy.bmat([
			self.l1.get_b(),
			self.constraints_b
		])).flatten()

	def get_shifted_polyhedron(self):
		A, b = self.get_polyhedron()
		# return A / self.l1.radius, b + numpy.dot(A, self.l1.center) / self.l1.radius
		return A * self.l1.radius, b - numpy.dot(A, self.l1.center)

	def add_shifted_pyomo_constraints(self, model):
		A, b = self.get_shifted_polyhedron()
		_add_constraints_to_pyomo(model, A, b)

	def add_unshifted_pyomo_constraints(self, model):
		A, b = self.get_polyhedron()
		_add_constraints_to_pyomo(model, A, b)

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
		A, b = self.get_shifted_polyhedron()
		ret = [numpy.zeros(len(self.l1.center))]
		while len(ret) + 1 < num_points:
			p = numpy.zeros(len(self.l1.center))
			for i in range(len(self.l1.center)):
				p[i] = 2 * numpy.random.random() - 1
			if (numpy.dot(A, p) < b).all():
				ret.append(p)
		return ret
