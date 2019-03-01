import numpy

from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.dfo.trust_region.circular_trust_region import CircularTrustRegion


def get_circular_trust_region_objective(context, x, hot_start, options):
	polyhedron = context.construct_polyhedron()
	distance_to_closest_constraint = polyhedron.distance_to_closest_constraint(x)
	distance_to_center = numpy.linalg.norm(x - context.model_center())

	value = ObjectiveValue()
	value.point = x
	value.success = distance_to_center <= distance_to_closest_constraint
	value.objective = distance_to_closest_constraint ** 2
	value.hot_start = None
	value.trust_region = CircularTrustRegion(
		center=x,
		radius=distance_to_closest_constraint
	)
	return value
