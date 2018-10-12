import numpy

from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.optimization.maximize_ellipse import EllipseParams
from trust_region.optimization.maximize_ellipse import compute_maximal_ellipse
from trust_region.optimization.scipy_maximize_ellipse import old_maximize_ellipse
from trust_region.util.plots import create_plot


# def get_pyomo_elliptical_trust_region_objective(context, x, hot_start, options):
# 		must_include_center = options['must_include_center']
#
# 		ellipse_params = EllipseParams()
# 		ellipse_params.center = x
# 		ellipse_params.A, ellipse_params.b = context.get_polyhedron()
# 		ellipse_params.include_point = numpy.copy(context.model_center()) if must_include_center else None
# 		ellipse_params.tolerance = context.params.subproblem_constraint_tolerance
# 		ellipse_params.hot_start = None  # hot_start
#
# 		value = ObjectiveValue()
# 		value.point = x
#
# 		if must_include_center:
# 			a, b = context.get_polyhedron()
# 			if (numpy.dot(a, x + (x - context.model_center())) > b).any():
# 				value.success = False
# 				value.trust_region = None
# 				return value
#
# 		try:
# 			value.success, value.trust_region = compute_maximal_ellipse(ellipse_params)
# 		except:
# 			value.success = False
# 			value.trust_region = None
# 			#ellipse_params.hot_start = None
# 			#value.success, value.trust_region = compute_maximal_ellipse(ellipse_params)
# 		if value.success:
# 			value.objective = value.trust_region.volume
# 			value.hot_start = value.trust_region.hot_start
# 		return value


difference_plot_count = 0


def get_elliptical_trust_region_objective(context, x, hot_start, options):
		must_include_center = options['must_include_center']

		ellipse_params = EllipseParams()
		ellipse_params.center = x
		ellipse_params.A, ellipse_params.b = context.get_polyhedron()
		ellipse_params.include_point = numpy.copy(context.model_center()) if must_include_center else None
		ellipse_params.tolerance = context.params.subproblem_constraint_tolerance
		ellipse_params.hot_start = None  # hot_start

		value = ObjectiveValue()
		value.point = x

		if must_include_center:
			a, b = context.get_polyhedron()
			if (numpy.dot(a, x + (x - context.model_center())) > b).any():
				value.success = False
				value.trust_region = None
				return value

		try:
			success1, ellipse1 = compute_maximal_ellipse(ellipse_params)
			#if success1:
			#	success1 = ellipse1.evaluate(context.model_center()) <= 1.0
		except:
			success1 = False
			ellipse1 = None
		try:
			success2, ellipse2 = old_maximize_ellipse(ellipse_params)
			#if success2:
			#	success2 = ellipse2.evaluate(context.model_center()) <= 1.0
		except:
			success2 = False
			ellipse2 = None

		if not success1 or not success2 or abs(ellipse1.volume - ellipse2.volume) > 1e-4:
			if success1:
				print(ellipse1.volume)
			if success2:
				print(ellipse2.volume)

			global difference_plot_count
			difference_plot_count += 1
			plot = create_plot(
				title='ellipses',
				filename='images/different_ellipses_{}.png'.format(str(difference_plot_count).zfill(4)),
				bounds=context.outer_trust_region.get_bounds().expand()
			)
			context.outer_trust_region.add_to_plot(plot)
			if success1:
				ellipse1.add_to_plot(plot, color='b', detailed=False)
			if success2:
				ellipse2.add_to_plot(plot, color='y', detailed=False)
			plot.ax.text(
				0.1, 0.1,
				'pyomo: ' + (str(ellipse1.volume) if success1 else 'invalid') + ' (blue), scipy: ' +
					(str(ellipse2.volume) if success2 else 'invalid') + ' (yellow)',
				horizontalalignment='center',
				verticalalignment='center',
				transform=plot.ax.transAxes
			)
			plot.save()
			print('large difference in solvers')

		value.hot_start = ellipse1.hot_start if success1 else None
		if (success2 and not success1) or (success2 and success1 and ellipse2.volume > ellipse1.volume):
			value.success = success2
			value.trust_region = ellipse2
			value.objective = ellipse2.volume
		else:
			value.success = success1
			value.trust_region = ellipse1
			value.objective = ellipse1.volume

		return value
