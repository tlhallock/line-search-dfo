import numpy
from trust_region.util.directions import sample_search_directions
from trust_region.algorithm.tr_search.searches.common import NoPlotDetails


def random_point(A, b, bounds):
	point = bounds.sample()
	while (numpy.dot(A, point) > b).any():
		point = bounds.sample()
	return point


def get_starting_points(A, b, x0, bounds, n):
	yield x0
	for _ in range(n):
		yield random_point(A, b, bounds)


def search_anywhere(context, objective, options):
	A, b = context.get_polyhedron()
	x0 = numpy.copy(context.model_center())
	bounds = context.outer_truster_region.get_bounds()
	starting_radius = options['starting_radius']

	best_solution_so_far = None

	for starting_point in get_starting_points(A, b, x0, bounds, 5):
		local_best_point = starting_point
		local_best_solution = objective(context=context, x=starting_point, hot_start=None, options=options)
		if not local_best_solution.success:
			continue

		radius = starting_radius
		while radius > tolerance:
			improved = False
			for search_direction in sample_search_directions(len(x0), 5):
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
				if trial_solution.objective > local_best_solution.objective:
					continue

				local_best_solution = trial_solution
				local_best_point = trial_point
				improved = True
				break

			if not improved:
				radius /= 2.0
		if best_solution_so_far is not None and local_best_solution.objective > best_solution_so_far.objective:
			continue

		best_solution_so_far = local_best_solution

	return best_solution_so_far, NoPlotDetails()
