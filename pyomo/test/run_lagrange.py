import numpy

import matplotlib.pyplot as plt
from trust_region.util.basis import QuadraticBasis
from trust_region.util.trust_region import CircularTrustRegion
from trust_region.dfo.lagrange import computeLagrangePolynomials
from trust_region.dfo.lagrange import LagrangeParams

points = numpy.asarray([
	[5.0, -2.5],
	[5.0, -2.5],
	[5.0, -2.5],
	[5.0, -2.5],
	[5.0, -2.5],
	[5.0, -2.5],
])


basis = QuadraticBasis(2)
params = LagrangeParams()
trust_region = CircularTrustRegion(
	numpy.asarray([5.0, -2.5]),
	3.0
)
context = {}

certification = computeLagrangePolynomials(basis, trust_region, points, context, params)

fig = plt.figure()
fig.set_size_inches(15, 15)
ax = fig.add_subplot(111)
trust_region.add_to_plot(ax)
certification.add_to_plot(ax)
plt.legend(loc='lower left')
fig.savefig('output.png')
plt.close()


print(certification.shifted)
print(certification.unshifted)

