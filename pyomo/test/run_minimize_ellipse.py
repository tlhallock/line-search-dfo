import numpy

from trust_region.util.plots import create_plot
from trust_region.optimization.minimize_circumscribed_ellipse import minimize_ellipse
from trust_region.util.bounds import Bounds
from trust_region.dfo.trust_region.ellipse import Ellipse


points = numpy.random.normal([2, 1], [3.0, 5.0], (6, 2))

ellipse = minimize_ellipse(points, 1e-4)

bounds = Bounds()
for p in points:
	bounds.extend(p)
bounds.extend(numpy.array([+10, +10]))
bounds.extend(numpy.array([-10, -10]))
plot = create_plot('an ellipse', 'images/circumscribed_ellipse.png', bounds.expand())
plot.add_points(points, 'the points')
ellipse.add_to_plot(plot)
plot.save()




# factor = 0.9
# q = ellipse.q
# c = ellipse.center
# improved = True
# while improved:
# 	improved = False
# 	w, d, v = numpy.linalg.svd(q)
# 	for i in range(len(d)):
# 		old_d = d[i]
# 		#d[i] *= factor
# 		print(d)
# 		q_trial = numpy.asarray(numpy.asmatrix(w) * numpy.asmatrix(numpy.diag(d)) * numpy.asmatrix(v))
#
# 		feasible = True
# 		for p in points:
# 			print(1 - numpy.dot(p - c, numpy.dot(q, p - c)))
# 			if 1 - 0.5 * numpy.dot(p - c, numpy.dot(q, p - c)) > 0:
# 				feasible = False
# 				break
#
# 		if feasible:
# 			improved = True
# 			q = q_trial
# 		else:
# 			d[i] = old_d
#
#
# ellipse2 = Ellipse()
# ellipse2.center = ellipse.center
# ellipse2.volume = None  # Can fill this in...
# ellipse2.ds = None
# ellipse2.lambdas = None
# ellipse2.q = q
# ellipse2.q_inverse = numpy.linalg.inv(q)
# ellipse2.l = numpy.linalg.cholesky(ellipse.q).T
# ellipse2.l_inverse = numpy.linalg.inv(ellipse.l)
# ellipse2.hot_start = None
#
# ellipse2.add_to_plot(plot)

