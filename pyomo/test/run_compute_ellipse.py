import numpy

from trust_region.optimization.maximize_ellipse import compute_maximal_ellipse
from trust_region.optimization.maximize_ellipse import EllipseParams
from trust_region.util.plots import create_plot
from trust_region.util.history import Bounds

# y <= 10 - 2x
# x >= 0
# y >= x


ellipse_params = EllipseParams()
#ellipse_params.A = numpy.array([
#	[1,  2],
#	[-1, 0],
#	[1, -1],
#])
ellipse_params.A = numpy.array(
	[[-1., 1.],
	 [-1., -1.],
	 [1., 0.],
	 [-1., 0.],
	 [0., 1.],
	 [0., -1.]]
)
#ellipse_params.b = numpy.array([
#	10,
#	0,
#	0,
#])
ellipse_params.b = numpy.array(
	[0., 0., 2.08855352, -0.08855352, 1.34809804, 0.65190196]
)


#ellipse_params.center = numpy.array([1, 2.5])
#ellipse_params.center = numpy.array([2.08518851,  0.33922567])
ellipse_params.center = numpy.array([1.28518851,  0.33922567])

ellipse_params.tolerance = 1e-12
#ellipse_params.include_point = numpy.array([2, 3])
ellipse_params.include_point = numpy.array([1.08855352,  0.34809804])


success, ellipse = compute_maximal_ellipse(ellipse_params)

if ellipse is not None:
	print(ellipse.volume)

print(success)

bounds = Bounds()
bounds.extend(numpy.array([3, 3]))
bounds.extend(numpy.array([-2, -2]))

plot = create_plot('testing_ellipse', 'images/ellipse.png', bounds)

plot.add_polyhedron(ellipse_params.A, ellipse_params.b, label='bounds')
plot.add_point(ellipse_params.center, label='center', marker='x', color='r')
plot.add_point(ellipse_params.include_point, label='include', marker='+', color='y')
if ellipse is not None:
	ellipse.add_to_plot(plot)
plot.save()




