
import numpy

import itertools
from trust_region.util.enumerate_polyhedron import enumerate_vertices_of_polyhedron
from trust_region.util.enumerate_polyhedron import get_polyhedron
from trust_region.util.bounds import Bounds
from trust_region.util.plots import create_plot


A = numpy.asarray([
	[-1, 3],
	[-2, -1],
	[0, -1],
	[1, -1],
	[3, 4],
])
b = numpy.asarray([14, 0, 0, 5, 36])



plot_count = 0
vertices = numpy.array([x for x, _ in enumerate_vertices_of_polyhedron(A, b)])
for i in range(3, 6):
	for indices in itertools.combinations(range(len(vertices)), i):
		A_sub, b_sub = get_polyhedron(vertices[indices, :])

		bounds = Bounds()
		bounds.extend(numpy.array([8, 6]))
		bounds.extend(numpy.array([-2.5, -1]))

		plot = create_plot('stuff', 'images/polyhedron_stuff_{}.png'.format(plot_count), bounds.expand())
		plot.add_polyhedron(A, b, 'the poly', color='k')
		plot.add_polyhedron(A_sub, b_sub, 'the poly', color='b')
		plot.save()
		plot_count += 1
