import numpy

from trust_region.optimization.maximize_ellipse import EllipseParams
from trust_region.optimization.maximize_ellipse import compute_maximal_ellipse


def get_elliptical_trust_region_objective(context, x, hot_start, options)
		must_include_center = options['must_include_center']

		ellipse_params = EllipseParams()
		ellipse_params.normalize = False
		ellipse_params.center = x
		ellipse_params.A, ellipse_params.b = context.get_polyhedron()
		ellipse_params.include_point = numpy.copy(context.model_center()) if must_include_center else None
		ellipse_params.tolerance = self.context.params.subproblem_constraint_tolerance

		value = ObjectiveValue()
		value.point = x
		value.success, value.trust_region = compute_maximal_ellipse(ellipse_params)
		value.objective = value.trust_region.volume
		return value
