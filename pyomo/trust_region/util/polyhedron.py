import itertools
import numpy
from trust_region.util.nullspace import nullspace


def get_hyperplane(vertices):
	n = nullspace(numpy.bmat([
		vertices,
		-numpy.asmatrix(numpy.ones(vertices.shape[0])).T
	]))
	A_h = n[:-1]
	b_h = n[len(n)-1]
	nrm = numpy.linalg.norm(A_h)
	return numpy.asarray(A_h / nrm).flatten(), b_h[0, 0] / nrm


def add_hyperplane(vertices, As, bs, A_h, b_h):
	for vertex in vertices:
		if numpy.dot(A_h, vertex) > b_h + 1e-10:
			return
	As.append(A_h)
	bs.append(b_h)


def get_polyhedron(vertices):
	dim = vertices.shape[1]
	As = []
	bs = []
	for indices in itertools.combinations(range(len(vertices)), dim):
		A_h, b_h = get_hyperplane(vertices[indices, :])
		add_hyperplane(vertices, As, bs, +A_h, +b_h)
		add_hyperplane(vertices, As, bs, -A_h, -b_h)
	return Polyhedron(numpy.array(As), numpy.array(bs))


def parse_polyhedron(json_object):
	return Polyhedron(
		numpy.array(json_object['A']),
		numpy.array(json_object['b'])
	)


class Polyhedron:
	def __init__(self, A, b):
		self.A = A.astype(numpy.float64)
		self.b = b.astype(numpy.float64)

	def evaluate(self, x):
		return self.b - numpy.dot(self.A, x)

	def add_single_constraint(self, a, b):
		return Polyhedron(
			numpy.vstack([self.A, a]),
			numpy.concatenate([self.b, [b]])
		)

	def add_lb(self, idx, c):
		return self.add_single_constraint(
			numpy.pad([-1.0], pad_width=[[idx, self.A.shape[1] - idx - 1]], mode='constant'),
			-c
		)

	def add_ub(self, idx, c):
		return self.add_single_constraint(
			numpy.pad([1.0], pad_width=[[idx, self.A.shape[1] - idx - 1]], mode='constant'),
			c
		)

	def distance_to_closest_constraint(self, x):
		return min(
			numpy.divide(abs(numpy.dot(self.A, x) - self.b), numpy.linalg.norm(self.A, axis=1))
		)

	def clone(self):
		return Polyhedron(numpy.copy(self.A), numpy.copy(self.b))

	def add_to_pyomo(self, model):
		for r in range(self.A.shape[0]):
			model.constraints.add(
				sum(model.x[c] * self.A[r, c] for c in model.dimension) <= self.b[r]
			)

	def contains(self, point, tolerance=1e-10):
		return (numpy.dot(self.A, point) <= self.b + tolerance).all()

	def shrink(self, center, factor):
		A = numpy.copy(self.A)
		b = numpy.copy(self.b)
		for i in range(A.shape[0]):
			n = numpy.linalg.norm(A[i])
			A[i] /= n
			b[i] /= n
		return Polyhedron(A, b * factor + (1-factor) * numpy.dot(A, center))

	def translate(A, b, center):
		return Polyhedron(numpy.copy(A), b + numpy.dot(A, center))

	def rotate(self, theta):
		if self.A.shape[1] != 2:
			raise Exception("Not supported")
		rotation = numpy.array([
			[+numpy.cos(theta), -numpy.sin(theta)],
			[+numpy.sin(theta), +numpy.cos(theta)]
		])
		return Polyhedron(numpy.dot(self.A, rotation), numpy.copy(self.b))

	def intersect(self, other):
		return Polyhedron(
			numpy.append(self.A, other.A, axis=0),
			numpy.append(self.b, other.b)
		)

	def enumerate_vertices(self):
		dimension = self.A.shape[1]
		num_constraints = self.A.shape[0]

		for indices in itertools.combinations(range(num_constraints), dimension):
			sub_a = self.A[list(indices), :]
			sub_b = self.b[list(indices)]

			try:
				x = numpy.linalg.solve(sub_a, sub_b)
			except numpy.linalg.LinAlgError:
				continue

			if not self.contains(x):
				continue

			yield x, indices

	def get_feasible_point(self, tolerance=1e-4):
		vertices = numpy.array([v[0] for v in self.enumerate_vertices()])
		central_point = numpy.mean(vertices, axis=0)
		if vertices.shape[0] < vertices.shape[1] + 1:
			directions = self.A[self.A@central_point > self.b - tolerance, :]
			direction = numpy.mean(numpy.diag(-1/numpy.linalg.norm(directions, axis=1)) @ directions, axis=0)
			direction /= numpy.linalg.norm(direction)
			t = 1
			while not self.contains(central_point + t * direction, tolerance):
				t /= 2
			if not self.contains(central_point + t * direction, tolerance):
				raise Exception("this algorithm did not work")
			return central_point + t * direction
		return central_point

	def get_diameter(self):
		diam = -1
		vertices = numpy.array([v for v in self.enumerate_vertices()])
		for idx1, idx2 in itertools.combinations(range(len(vertices)), 2):
			d = numpy.linalg.norm(vertices[idx1] - vertices[idx2])
			if d > diam:
				diam = d
		return diam

	def shift(self, center, radius):
		return Polyhedron(
			self.A * radius,
			self.b - numpy.dot(self.A, center)
		)

	def to_json(self):
		return {
			'A': self.A,
			'b': self.b
		}

	def normalize(self):
		A = self.A
		b = self.b
		for i in range(A.shape[0]):
			row_scale = 1.0 / abs(b[i])
			if numpy.isinf(row_scale) or numpy.isnan(row_scale) or row_scale < 1e-12:
				row_scale = 1.0 / numpy.linalg.norm(A[i])
			A[i] *= row_scale
			b[i] *= row_scale
		return Polyhedron(A, b)
