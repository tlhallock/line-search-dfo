
import numpy
import math
import itertools

from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.dfo.trust_region.polyhedral_trust_region import PolyhedralTrustRegion


def simplex_volume(points):
	n = points.shape[1]
	D = numpy.zeros((n, n))
	for i in range(n):
		D[:, i] = points[i] - points[n]
	return abs(numpy.linalg.det(D)) / math.factorial(n)


def get_scaled_simplex_trust_region_objective(context, x, hot_start, options):
	value = ObjectiveValue()
	value.point = None
	value.trust_region = PolyhedralTrustRegion(context.outer_trust_region, x[0], x[1])
	value.success = True
	value.objective = simplex_volume(x[2])
	n = x[0].shape[1]

	value.points_hint = numpy.zeros((int(1 + n + n*(n+1)/2), n))
	idx = 0
	# value.points_hint[idx] = context.model_center()
	for vertex in x[2]:
		value.points_hint[idx] = vertex
		idx += 1
	for v1, v2 in itertools.combinations(range(len(x[2])), 2):
		value.points_hint[idx] = 0.5 * (x[2][v1] + x[2][v2])
		idx += 1
	return value

