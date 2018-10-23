
import numpy
import itertools

from trust_region.algorithm.tr_search.searches.common import NoPlotDetails
from trust_region.util.enumerate_polyhedron import enumerate_vertices_of_polyhedron
from trust_region.util.enumerate_polyhedron import get_polyhedron
from trust_region.util.bounds import Bounds
from trust_region.util.plots import create_plot


class SimplexSearchPlot:
	def __init__(self):
		self.center = None
		self.outer_polyhedron = None
		self.inner_simplices = []
		self.points_hint = None
		self.best_polyhedron = None
		self.volume = None

	def add_to_plot(self, plot):
		plot.add_point(self.center, 'center', color='b')
		for a, b in self.inner_simplices:
			plot.add_polyhedron(a, b, label='inner', color='y', lvls=[0])
		a, b = self.outer_polyhedron
		plot.add_polyhedron(a, b, label='outer', color='r', lvls=[0])
		a, b = self.best_polyhedron
		plot.add_polyhedron(a, b, label='outer', color='g', lvls=[0])
		plot.add_points(self.points_hint, label='hints', color='m')
		plot.ax.text(
			0.2, 0.2,
			'simplex volume: ' + str(self.volume),
			horizontalalignment='center',
			verticalalignment='center',
			transform=plot.ax.transAxes
		)


def search_simplices(context, objective, options):
	A, b = context.get_polyhedron()
	x0 = numpy.copy(context.model_center())
	vertices = numpy.array([v for v, _ in enumerate_vertices_of_polyhedron(A, b)])
	best_solution_so_far = None

	search_plot = SimplexSearchPlot()
	search_plot.center = x0
	search_plot.outer_polyhedron = A, b
	for indices in itertools.combinations(range(len(vertices)), len(x0) + 1):
		sub_vertices = vertices[indices, :]
		A_sub, b_sub = get_polyhedron(sub_vertices)
		if (numpy.dot(A_sub, x0) > b_sub + context.params.subproblem_search_tolerance).any():
			continue
		search_plot.inner_simplices.append((A_sub, b_sub))
		trial_solution = objective(context, (A_sub, b_sub, sub_vertices), None, options)
		if best_solution_so_far is None or trial_solution.objective > best_solution_so_far.objective:
			best_solution_so_far = trial_solution

			search_plot.best_polyhedron = A_sub, b_sub
			search_plot.volume = trial_solution.objective
			search_plot.points_hint = trial_solution.points_hint

	bounds = Bounds()
	for p in vertices:
		bounds.extend(p)
	plot = create_plot('simplex_search', 'images/{}/{}_simplex_search.png'.format(
		context.params.directory,
		str(context.plot_number).zfill(5)
	), bounds.expand())
	context.plot_number += 1
	search_plot.add_to_plot(plot)
	plot.save()

	return best_solution_so_far, NoPlotDetails()
