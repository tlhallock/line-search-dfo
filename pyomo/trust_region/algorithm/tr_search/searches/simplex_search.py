
import numpy
import itertools

from trust_region.algorithm.tr_search.searches.common import NoPlotDetails
from trust_region.util.polyhedron import get_polyhedron
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
		for p in self.inner_simplices:
			plot.add_polyhedron(p, label='inner', color='y', lvls=[0])
		plot.add_polyhedron(self.outer_polyhedron, label='outer', color='r', lvls=[0])
		plot.add_polyhedron(self.best_polyhedron, label='outer', color='g', lvls=[0])
		plot.add_points(self.points_hint, label='hints', color='m')
		plot.ax.text(
			0.2, 0.2,
			'simplex volume: ' + str(self.volume),
			horizontalalignment='center',
			verticalalignment='center',
			transform=plot.ax.transAxes
		)


def search_simplices(context, objective, options):
	polyhedron = context.construct_polyhedron()
	x0 = numpy.copy(context.model_center())
	vertices = numpy.array([v for v, _ in polyhedron.enumerate_vertices()])
	best_solution_so_far = None

	search_plot = SimplexSearchPlot()
	search_plot.center = x0
	search_plot.outer_polyhedron = polyhedron
	for indices in itertools.combinations(range(len(vertices)), len(x0) + 1):
		sub_vertices = vertices[indices, :]
		sub_polyhedron = get_polyhedron(sub_vertices)
		if not sub_polyhedron.contains(x0, -context.params.subproblem_search_tolerance):
			continue
		search_plot.inner_simplices.append(sub_polyhedron)
		trial_solution = objective(context, (sub_polyhedron, sub_vertices), None, options)
		if best_solution_so_far is None or trial_solution.objective > best_solution_so_far.objective:
			best_solution_so_far = trial_solution

			search_plot.best_polyhedron = sub_polyhedron
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
