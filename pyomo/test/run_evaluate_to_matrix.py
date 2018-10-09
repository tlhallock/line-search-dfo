import numpy

from trust_region.basis import QuadraticBasis


points = numpy.asarray([
	[0.0, 0.0],
	[1.0, 0.0],
	[0.0, 1.0],
	[1.0, 1.0],
	[-1.0, -1.0]
])

q = QuadraticBasis(2)
print(q.evaluate_to_matrix(points))