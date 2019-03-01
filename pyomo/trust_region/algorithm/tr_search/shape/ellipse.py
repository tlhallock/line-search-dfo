import numpy

from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.optimization.maximize_multidimensional_ellipse import EllipseParams
from trust_region.optimization.maximize_multidimensional_ellipse import compute_maximal_ellipse_after_shift


def get_elliptical_trust_region_objective(context, x, hot_start, options):
		must_include_center = options['must_include_center']

		ellipse_params = EllipseParams()
		ellipse_params.center = x
		ellipse_params.polyhedron = context.construct_polyhedron()
		ellipse_params.include_point = numpy.copy(context.model_center()) if must_include_center else None
		ellipse_params.tolerance = context.params.subproblem_constraint_tolerance
		ellipse_params.hot_start = None  # hot_start

		value = ObjectiveValue()
		value.point = x

		if must_include_center:
			if (numpy.dot(ellipse_params.polyhedron.A, x + (x - context.model_center())) > ellipse_params.polyhedron.b).any():
				value.success = False
				value.trust_region = None
				return value

		value.success, value.trust_region = compute_maximal_ellipse_after_shift(ellipse_params, context.outer_trust_region)
		if value.success:
			value.objective = value.trust_region.volume
			value.hot_start = value.trust_region.hot_start

		if 'evaluator' in options:
			value.objective = options['evaluator'](
				indicator=lambda s: value.trust_region.contains(s)
			)

		return value
