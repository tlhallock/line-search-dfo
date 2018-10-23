import numpy
from trust_region.dfo.trust_region.ellipse import Ellipse


# Taken from
# https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
def minimize_circumscribed_ellipse(P, tolerance):
	d, N = P.shape
	Q = numpy.ones((d+1, N))
	Q[:-1] = P

	count = 1
	err = 1
	u = (1/N) * numpy.ones(N)

	while err > tolerance:
		X = numpy.dot(numpy.dot(Q, numpy.diag(u)), Q.T)
		M = numpy.diag(numpy.dot(numpy.dot(Q.T, numpy.linalg.inv(X)), Q))
		j = numpy.argmax(M)
		maximum = M[j]
		step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
		new_u = (1 - step_size) * u
		new_u[j] += step_size
		count += 1
		err = numpy.linalg.norm(new_u - u)
		u = new_u

	U = numpy.diag(u)
	A = (1 / d) * numpy.linalg.inv(
		numpy.dot(numpy.dot(P, U), P.T) -
		numpy.asmatrix(numpy.dot(P, u)).T * numpy.asmatrix(numpy.dot(P, u))
	)
	c = numpy.dot(P, u)
	return numpy.asarray(A), c


def minimize_ellipse(points, tol):
	Q, c = minimize_circumscribed_ellipse(points.T, tol)
	ellipse = Ellipse()

	ellipse.center = c
	ellipse.volume = numpy.pi / numpy.sqrt(numpy.linalg.det(Q))
	ellipse.ds = None
	ellipse.lambdas = None
	ellipse.q = 2 * Q
	ellipse.q_inverse = numpy.linalg.inv(2 * Q)
	ellipse.l = numpy.linalg.cholesky(ellipse.q).T
	ellipse.l_inverse = numpy.linalg.inv(ellipse.l)
	ellipse.hot_start = None

	return ellipse
