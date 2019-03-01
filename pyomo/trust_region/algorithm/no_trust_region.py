
import numpy

from trust_region.util.plots import create_plot
from trust_region.util.bounds import Bounds
from trust_region.util.polyhedron import get_polyhedron
from trust_region.util.polyhedron import get_diameter


# distance to expected minimum
# accuracy of expected minimum
# given a set of p points
#
# likelihood of points being critical points
#
# minimum value, lowest criticality, error bounds in the lowest criticality
#
# \nabla m(x)
# e(x) = min( f_i + |x_i - x| L )
#
#
# (x,y)
#

def objective(x):
	return 0.001 * (x[0] ** 2 + x[1] ** 2) + numpy.sin(numpy.sqrt((x[0] ** 2 + x[1] ** 2)))


class State:
	def __init__(self):
		self.L = 2
		self.kappa = 0.5
		self.points = None
		self.values = None
		self.A = None
		self.b = None
		self.diam = None
		self.phi = None

	def set_points(self, points):
		self.points = points
		self.values = numpy.array([objective(x) for x in self.points])
		self.A, self.b = get_polyhedron(self.points)
		self.diam = get_diameter(self.A, self.b)
		self.phi = numpy.asarray(numpy.linalg.solve(
			numpy.array([
				self.eval_basis(p)
				for p in self.points
			]),
			self.values
		)).flatten()

	def e(self, x):
		trial_error = numpy.min([
			self.values[i] + self.L * numpy.linalg.norm(self.points[i] - x)
			for i in range(self.points.shape[0])
		])

		if (numpy.dot(self.A, x) < self.b).all():
			if self.diam * self.kappa < trial_error:
				trial_error = self.diam * self.kappa

		return trial_error

	def eval_basis(self, x):
		return numpy.array([1.0, x[0], x[1], 0.5 * x[0] * x[0], 0.5 * x[0] * x[1], 0.5 * x[1] * x[1]])

	def eval_gradient(self, x):
		return numpy.array([
			[0.0, 0.0],
			[1.0, 0.0],
			[0.0, 1.0],
			[x[0], 0.0],
			[0.5 * x[1], 0.5 * x[0]],
			[0.0, x[1]]
		])

	def get_model(self):
		return (
			lambda x: numpy.dot(self.phi, self.eval_basis(x)),
			lambda x: numpy.dot(self.eval_gradient(x).T, self.phi)
		)


MAG = 2

base_points = MAG * (2 * numpy.random.random((6, 2)) - 1)
state = State()


iteration = 0
while True:
	iteration += 1

	base_points[0] = MAG * (2 * numpy.random.random(2) - 1)

	state.set_points(base_points)
	m, g = state.get_model()

	bounds = Bounds()
	bounds.extend(numpy.array([-MAG, -MAG]))
	bounds.extend(numpy.array([+MAG, +MAG]))
	plot = create_plot('foobar', 'images2/model_{}.png'.format(str(iteration).rjust(4, '0')), bounds.expand())

	#plot.add_contour(lambda x: objective(x), 'objective', color='g')
	plot.add_contour(lambda x: state.e(x), 'error', color='r')
	#plot.add_contour(lambda x: m(x), 'model', color='y')
	#plot.add_contour(lambda x: numpy.linalg.norm(g(x)), 'error', color='k')
	#plot.add_points(state.points, 'sample points', color='m')
	A, b = get_polyhedron(state.points)
	#plot.add_polyhedron(A, b, label='poly', color='c')

	for p in state.points:
		if abs(state.get_model()[0](p) - objective(p)) > 1e-4:
			raise Exception('bug')

	plot.save()
