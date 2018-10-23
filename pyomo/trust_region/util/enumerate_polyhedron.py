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
	return numpy.array(As), numpy.array(bs)


def enumerate_vertices_of_polyhedron(A, b):
	dimension = A.shape[1]
	num_constraints = A.shape[0]

	for indices in itertools.combinations(range(num_constraints), dimension):
		sub_a = A[list(indices), :]
		sub_b = b[list(indices)]

		try:
			x = numpy.linalg.solve(sub_a, sub_b)
		except numpy.linalg.LinAlgError:
			continue

		if (numpy.dot(A, x) > b + 1e-10).any():
			continue

		yield x, indices


def get_diameter(A, b):
	diam = -1
	vertices = numpy.array([v for v in enumerate_vertices_of_polyhedron(A, b)])
	for idx1, idx2 in itertools.combinations(range(len(vertices)), 2):
		d = numpy.linalg.norm(vertices[idx1] - vertices[idx2])
		if d > diam:
			diam = d
	return diam