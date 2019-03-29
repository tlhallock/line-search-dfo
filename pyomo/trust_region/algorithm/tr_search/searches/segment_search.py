import numpy
from trust_region.algorithm.tr_search.searches.segment_search_path import get_search_path


def search_segment(context, objective, options):
	polyhedron = context.construct_polyhedron()
	x0 = numpy.copy(context.model_center())
	tolerance = context.params.subproblem_search_tolerance * context.outer_trust_region.radius
	number_of_points = options['number_of_points']
	num_trial_points = options['num_trial_points']

	search_path = get_search_path(x0, polyhedron, number_of_points, 1e-12)
	best_solution_so_far = objective(
		context=context,
		x=x0,
		hot_start=None,
		options=options
	)

	plot_count = 0

	delta = 0.5
	center = 0.5
	while delta > tolerance and len(search_path.points) > 1:
		for t in numpy.linspace(center - delta, center + delta, num_trial_points):
			other_point = search_path.get_point(t)
			if not polyhedron.contains(other_point, 1e-6):
				continue
				# raise Exception('search path includes infeasible points')

			other_solution = objective(
					context=context,
					x=other_point,
					hot_start=best_solution_so_far.hot_start,
					options=options
				)
			if not other_solution.success:
				continue

			if best_solution_so_far.success and other_solution.objective <= best_solution_so_far.objective:
				continue

			# # #####################################################################################
			# from trust_region.util.history import Bounds
			# from trust_region.util.plots import create_plot
			#
			# bounds = Bounds()
			# bounds.extend(numpy.array([3, 3]))
			# bounds.extend(numpy.array([-2, -2]))
			#
			# plot = create_plot('testing_ellipse', 'images/ellipse_{}.png'.format(str(plot_count).zfill(4)), bounds)
			# plot_count += 1
			#
			# plot.ax.text(
			# 	0.1, 0.1,
			# 	str(other_solution.trust_region.volume),
			# 	horizontalalignment='center',
			# 	verticalalignment='center',
			# 	transform=plot.ax.transAxes
			# )
			#
			# plot.add_polyhedron(A, b, label='bounds')
			# plot.add_point(other_point, label='center', marker='x', color='r')
			# plot.add_point(context.model_center(), label='include', marker='+', color='y')
			# other_solution.trust_region.add_to_plot(plot)
			# plot.save()
			# # #####################################################################################

			best_solution_so_far = other_solution
			center = t

		delta = min(
			delta / (2 * num_trial_points),  # end points evaluated twice
			center,
			1 - center
		)

	return best_solution_so_far, search_path

