import numpy
from trust_region.util.directions import sample_search_directions
from trust_region.algorithm.tr_search.searches.common import NoPlotDetails
from trust_region.algorithm.tr_search.searches.common import ObjectiveValue


def random_point(A, b, bounds):
	point = bounds.sample()
	while (numpy.dot(A, point) > b).any():
		point = bounds.sample()
	return point


def get_starting_points(A, b, x0, bounds, n):
	yield x0
	for _ in range(n):
		yield random_point(A, b, bounds)


plot_count = 0
def search_anywhere(context, objective, options):
	global plot_count

	A, b = context.get_polyhedron()
	x0 = numpy.copy(context.model_center())
	bounds = context.outer_trust_region.get_bounds()
	tolerance = context.params.subproblem_search_tolerance * context.outer_trust_region.radius
	num_starting_points = options['random-starting-points']
	num_search_directions = options['random-search-directions']

	starting_point_count = 0
	best_solution_so_far = ObjectiveValue()
	for starting_point in get_starting_points(A, b, x0, bounds, num_starting_points):
		local_best_point = starting_point
		starting_point_count += 1
		# print('\tinner iteration {} of {}'.format(starting_point_count, num_starting_points + 1))

		local_best_solution = objective(context=context, x=starting_point, hot_start=None, options=options)
		if not local_best_solution.success:
			continue

		radius = context.outer_trust_region.radius / 2
		number_of_improvements = 0
		while radius > tolerance:
			# print('\t\t', starting_point_count, radius, local_best_solution.objective)

			improved = False
			for search_direction in sample_search_directions(len(x0), num_search_directions):
				trial_point = local_best_point + radius * search_direction
				if (numpy.dot(A, trial_point) > b).any():
					continue
				trial_solution = objective(
					context=context,
					x=trial_point,
					hot_start=local_best_solution.hot_start,
					options=options
				)
				if not trial_solution.success:
					continue
				if trial_solution.objective <= local_best_solution.objective:
					continue

				#####################################################################################
				#
				# from trust_region.util.history import Bounds
				# from trust_region.util.plots import create_plot
				#
				# bounds = Bounds()
				# bounds.extend(numpy.array([7, 3]))
				# bounds.extend(numpy.array([-2, -2]))
				#
				# plot = create_plot('expected ellipse value', 'images/ellipse_{}.png'.format(str(plot_count).zfill(4)), bounds)
				# plot_count += 1
				#
				# plot.ax.text(
				# 	0.1, 0.1,
				# 	str(trial_solution.objective) + ',' + str(trial_solution.trust_region.volume),
				# 	horizontalalignment='center',
				# 	verticalalignment='center',
				# 	transform=plot.ax.transAxes
				# )
				#
				# if 'value-of-point' in options:
				# 	plot.add_contour(options['value-of-point'], 'value', color='g')
				# plot.add_polyhedron(A, b, label='bounds')
				# plot.add_point(trial_point, label='center', marker='x', color='r')
				# plot.add_point(context.model_center(), label='include', marker='+', color='y')
				# trial_solution.trust_region.add_to_plot(plot)
				# plot.save()
				# #####################################################################################

				local_best_solution = trial_solution
				local_best_point = trial_point
				improved = True
				number_of_improvements += 1
				break

			if not improved or number_of_improvements > 10:
				radius /= 2.0
				number_of_improvements = 0
		if best_solution_so_far.objective is not None and\
			local_best_solution.objective <= best_solution_so_far.objective:
			continue

		best_solution_so_far = local_best_solution

	return best_solution_so_far, NoPlotDetails()
