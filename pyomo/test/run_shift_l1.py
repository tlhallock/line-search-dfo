import numpy

from trust_region.util.plots import create_plot
from trust_region.util.bounds import Bounds
from trust_region.dfo.trust_region.l1_trust_region import L1TrustRegion


l1 = L1TrustRegion(numpy.array([5, 1.5]), 2.0)

bounds = Bounds()
bounds.extend(numpy.array([+10, +10]))
bounds.extend(numpy.array([-10, -10]))

# #  add the constraints
#
#
# A, b = l1.get_polyhedron()
# A_s, b_s = l1.get_shifted...
# # A_s, b_s = A / self.l1.radius, b + numpy.dot(A, self.l1.center) / self.l1.radius
# A_s, b_s = A * self.l1.radius, b - numpy.dot(A, self.l1.center)
#
# plot = create_plot('testing', 'images/testing.png', bounds)
# plot.add_polyhedron(A, b, label='original', color='g')
# plot.add_polyhedron(A_s, b_s, label='original', color='r')
# plot.save()