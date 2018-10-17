import itertools
import numpy


def enumerate_vertices_of_polyhedron(A, b):
	dimension = A.shape[1]
	num_constraints = A.shape[0]
	points = []

	for indices in itertools.combinations([_ for _ in range(num_constraints)], dimension):
		sub_a = A[indices, :]
		sub_b = b[indices, :]

		try:
			x = numpy.linalg.solve(sub_a, sub_b)
		except numpy.linalg.LinAlgError:
			continue

		if (numpy.dot(A, x) > b).any():
			continue

		points.append(x)

	return points
