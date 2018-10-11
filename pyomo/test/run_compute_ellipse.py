import numpy

from trust_region.optimization.maximize_ellipse import compute_maximal_ellipse
from trust_region.optimization.maximize_ellipse import EllipseParams
from trust_region.util.plots import create_plot
from trust_region.util.history import Bounds

# y <= 10 - 2x
# x >= 0
# y >= x


ellipse_params = EllipseParams()
ellipse_params.A = numpy.array([
	[1,  2],
	[-1, 0],
	[1, -1],
])
ellipse_params.b = numpy.array([
	10,
	0,
	0,
])
ellipse_params.center = numpy.array([1.0, 2])

ellipse_params.normalize = False
ellipse_params.include_point = numpy.array([1.0, 2.5])

ellipse = compute_maximal_ellipse(ellipse_params)

bounds = Bounds()
bounds.lbX = -2
bounds.ubX = 7
bounds.lbY = -2
bounds.ubY = 7

plot = create_plot('testing_ellipse', 'images/ellipse.png', bounds)

plot.add_polyhedron(ellipse_params.A, ellipse_params.b, label='bounds')
plot.add_point(ellipse_params.center, label='center', marker='x', color='r')
plot.add_point(ellipse_params.include_point, label='center', marker='+', color='y')
ellipse.add_to_plot(plot)
plot.save()




