import numpy
import traceback

from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.optimization.maximize_2d_ellipse import EllipseParams
from trust_region.optimization.maximize_2d_ellipse import compute_maximal_ellipse_after_shift
from trust_region.dfo.trust_region.scaled_ellipse import ScaledEllipse


def get_scaled_elliptical_trust_region_objective(context, x, hot_start, options):
		ellipse_params = EllipseParams()
		ellipse_params.center = x
		ellipse_params.A, ellipse_params.b = context.get_polyhedron()
		ellipse_params.include_point = None
		ellipse_params.tolerance = context.params.subproblem_constraint_tolerance
		ellipse_params.hot_start = hot_start

		value = ObjectiveValue()
		value.point = x

		try:
			success, ellipse = compute_maximal_ellipse_after_shift(ellipse_params, context.outer_trust_region)
		except:
			traceback.print_exc()
			success = False
			ellipse = None

		value.success = success
		if not success:
			return value

		scale_value = ellipse.get_scale_to_include(context.model_center())
		A, b = context.get_polyhedron()

		value.hot_start = ellipse.hot_start
		value.trust_region = ScaledEllipse(ellipse, scale_value, A, b)
		value.objective = ellipse.volume
		return value
