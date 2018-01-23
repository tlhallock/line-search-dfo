

from dfo import polynomial_basis
from dfo import lagrange
from numpy import array as arr
from numpy.random import random
from numpy import zeros

n = 2
degree = 2

basis = polynomial_basis.PolynomialBasis(n, degree)

radius = 1
center = arr([0, 0])
poisedSet = center + zeros((basis.basis_dimension, n))
poisedSet = center + 0 * (2 * (.1 * random((basis.basis_dimension, n))) - 1)

print(poisedSet)

poisedSet[1] = poisedSet[0]
poisedSet[2] = poisedSet[0]


params = lagrange.LagrangeParams(center, radius, True, initialXsi=1e-5, consOpts=None)
params.improve = True
cert = lagrange.computeLagrangePolynomials(basis, poisedSet, params)

print(cert.poised)
print(cert.lmbda)
print(cert.indices)
print(cert.shifted)
print(cert.unshifted)

cert.plot('images/poised.png', center, radius)

