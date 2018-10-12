import numpy
from trust_region.algorithm.tr_search.searches.segment_search_path import get_search_path


def search_segment(context, objective, options):
	A, b = context.get_polyhedron()
	x0 = numpy.copy(context.model_center())
	tolerance = context.params.subproblem_tolerance
	number_of_points = options['number_of_points']
	num_trial_points = options['num_trial_points']

	search_path = get_search_path(x0, A, b, number_of_points)
	best_solution_so_far = objective(
		context=context,
		x=x0,
		hot_start=None,
		options=options
	)

	delta = 0.5
	center = 0.5
	while delta > tolerance and len(search_path.points) > 1:
		for t in numpy.linspace(center - delta, center + delta, num_trial_points):
			other_point = search_path.get_point(t)
			if (numpy.dot(A, other_point) > b).any():
				raise Exception('search path includes infeasible points')

			other_solution = objective(
					context=context,
					x=trial_point,
					hot_start=best_solution_so_far.hot_start,
					options=options
				)
			if not other_solution.success:
				continue

			if other_solution.objective > best_solution_so_far.objective or not best_solution_so_far.success:
				continue

			best_solution_so_far = other_solution
			center = t

		delta = min(
			delta / (2 * num_trial_points),  # end points evaluated twice
			center,
			1 - center
		)

	return best_solution_so_far, search_path

