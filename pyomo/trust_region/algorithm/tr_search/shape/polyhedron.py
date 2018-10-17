
import numpy

from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.dfo.trust_region.polyhedral_trust_region import PolyhedralTrustRegion


def get_polyhedral_trust_region_objective(context, x, hot_start, options):
	value = ObjectiveValue()
	value.point = x
	value.success = True
	value.objective = 1
	value.hot_start = None
	value.trust_region = PolyhedralTrustRegion(
		context.outer_trust_region,
		context.params.constraints_A,
		context.params.constraints_b
	)
	return value
