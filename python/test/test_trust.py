

from oct2py import octave
import os
from numpy import asmatrix
from utilities import trust
from numpy import array
from numpy.linalg import norm
from numpy import random


octave.addpath(os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../utilities/'))


def testIt(g, H, delta):
	s1, val1, posdef1, count1, lmbda1 = trust.trust(g, H, delta)
	s2, val2, posdef2, count2, lmbda2 = octave.trust(g, H, delta)

	if True:
		print(s1)
		print(s2)
		print(val1)
		print(val2)
		print(posdef1)
		print(posdef2)
		print(count1)
		print(count2)
		print(lmbda1)
		print(lmbda2)
		print('---------------')

	if norm(s1 - s2) > 1e-4:
		return False
	if norm(val1 - val2) > 1e-4:
		return False
	if not posdef1 == (posdef2 == 1):
		return False
#	if abs(count1 - count2) > 1.5: # They can be off by one
#		return False
	if norm(lmbda1 - lmbda2) > 1e-4:
		return False
	return True


def test(g, H, delta):
	if not testIt(g, H, delta):
		raise Exception('failed')

# indefinite
H = asmatrix(array([[1, 0], [0, -2]]))
g = asmatrix(array([1, 1])).T
delta = 1
test(g, H, delta)

# positive definite, repeated eigenvalues
H = asmatrix(array([[1, 0], [0, 1]]))
g = asmatrix(array([1, 1])).T
delta = 1
test(g, H, delta)

# positive definite
H = asmatrix(array([[1, 0], [0, 2]]))
g = asmatrix(array([1, 1])).T
delta = 1
test(g, H, delta)

# negative definite
H = asmatrix(array([[-1, 0], [0, -2]]))
g = asmatrix(array([1, 1])).T
delta = 1
test(g, H, delta)

# negative semi definite singular
H = asmatrix(array([[-1, 0], [0, 0]]))
g = asmatrix(array([1, 1])).T
delta = 1
test(g, H, delta)

# positive semi definite singular
H = asmatrix(array([[1, 0], [0, 0]]))
g = asmatrix(array([1, 1])).T
delta = 1
test(g, H, delta)

# entirely singular (line)
H = asmatrix(array([[0, 0], [0, 0]]))
g = asmatrix(array([1, 1])).T
delta = 1
test(g, H, delta)

# highly symmetric
H = asmatrix(array([[1, 0], [0, 1]]))
g = asmatrix(array([0, 0])).T
delta = 1
test(g, H, delta)

# constant function
H = asmatrix(array([[0, 0], [0, 0]]))
g = asmatrix(array([0, 0])).T
delta = 1
test(g, H, delta)

# another test that this failed at one point
H = asmatrix(array([[ 0.,  0.],[ 0.,  0.]]))
g = asmatrix(array([0, 1])).T
delta = 1
test(g, H, delta)

# another test that this failed at one point
H = asmatrix(array([[ -1.80000000e-01,  5.72875081e-14],[  5.72875081e-14,  -1.80000000e-01]]))
g = asmatrix(array([-4.63795669e-13, 0.00000000e+00])).T
delta = 1
test(g, H, delta)


for i in range(1000):
	print(i)

	n = random.randint(2, 10)
	H = 2 * asmatrix(random.random([n, n])) - 1
	H = H + H.T
	g = 2 * asmatrix(random.random([n, 1])) - 1
	delta = 5 * random.random()
	test(g, H, delta)
