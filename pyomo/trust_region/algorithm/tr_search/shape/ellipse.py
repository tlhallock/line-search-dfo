import numpy

from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.optimization.maximize_ellipse import EllipseParams
from trust_region.optimization.maximize_ellipse import compute_maximal_ellipse


def get_elliptical_trust_region_objective(context, x, hot_start, options):
		must_include_center = options['must_include_center']

		ellipse_params = EllipseParams()
		ellipse_params.center = x
		ellipse_params.A, ellipse_params.b = context.get_polyhedron()
		ellipse_params.include_point = numpy.copy(context.model_center()) if must_include_center else None
		ellipse_params.tolerance = context.params.subproblem_constraint_tolerance
		ellipse_params.hot_start = hot_start

		value = ObjectiveValue()
		value.point = x
		try:
			value.success, value.trust_region = compute_maximal_ellipse(ellipse_params)
		except:
			value.success = False
			value.trust_region = None
			#ellipse_params.hot_start = None
			#value.success, value.trust_region = compute_maximal_ellipse(ellipse_params)
		if value.success:
			value.objective = value.trust_region.volume
			value.hot_start = value.trust_region.hot_start
		return value
