
import numpy
import itertools
from trust_region.util.bounds import Bounds
from trust_region.dfo.trust_region.circular_trust_region import CircularTrustRegion


class L1TrustRegion(CircularTrustRegion):
	def copy(self):
		return L1TrustRegion(numpy.copy(self.center), self.radius)

	def add_to_plot(self, plot_object, detailed=True):
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
		for idx in range(len(self.center)):
			model.constraints.add(
				model.x[idx] <= +1.0
			)
			model.constraints.add(
				model.x[idx] >= -1.0
			)

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

	def get_bounds(self):
		bounds = Bounds()
		for i in range(len(self.center)):
			point = numpy.copy(self.center)
			point[i] = self.center[i] + self.radius
			bounds.extend(point)

			point = numpy.copy(self.center)
			point[i] = self.center[i] - self.radius
			bounds.extend(point)
		return bounds

	def sample_shifted_region(self, num_points):
		dim = len(self.center)
		ret = [numpy.zeros(dim)]
		for _ in range(num_points):
			p = numpy.zeros(dim)
			for i in range(dim):
				# p[i] = self.center[i] + self.radius * (2 * numpy.random.random() - 1)
				p[i] = 2 * numpy.random.random() - 1
			ret.append(p)
		return ret

	def sample_unshifted_region(self, num_points):
		dim = len(self.center)
		yield numpy.copy(self.center)
		for _ in range(num_points):
			p = numpy.copy(self.center)
			for i in range(dim):
				p[i] += self.radius * (2 * numpy.random.random() - 1)
			yield p

	def endpoints(self):
		for x in itertools.product([self.radius, -self.radius], repeat=len(self.center)):
			yield self.center + numpy.array(x)

	def contained_in(self, A, b):
		for endpoint in self.endpoints():
			if (numpy.dot(A, endpoint) > b).any():
				return False
		return True

	def contains(self, point):
		delta = point - self.center
		for d in delta:
			if d < -self.radius:
				return False
			if d > +self.radius:
				return False
		return True

	# def shift_polyhedron(self, polyhedron):
	#	 pass

