
import numpy


from trust_region.dfo.trust_region.ellipse import Ellipse
from scipy.optimize import minimize


def old_maximize_ellipse(p):
	A = p.A
	b = p.b
	include = p.include_point
	xbar = p.center
	tol  = p.tolerance

	bbar = b - numpy.dot(A, xbar.T)
	k = 1 / min(abs(bbar))
	bbar = k * bbar

	vec2mat = lambda q: numpy.asarray([[q[0], q[1]], [q[1], q[2]]])
	vec2det = lambda q: q[0] * q[2] - q[1] * q[1]
	vec2jac = lambda q: numpy.asarray([q[2], -2 * q[1], q[0]])
	def idx2cons(idx):
		Ar = A[idx,:]
		br = bbar[idx]
		def val(x):
			q = vec2mat(x)
			return -(numpy.dot(Ar, numpy.dot(q, Ar)) - br * br / 2)
		def jac(q):
			# [A[idx, 1], A[idx, 2]] * [[q[0], q[1]], [q[1], q[2]]] * [A[idx, 1], A[idx, 2]]
			# [A[idx, 1], A[idx, 2]] * [q[0] * A[idx, 1] + q[1] * A[idx, 2], q[1] * A[idx, 1] + q[2] * A[idx, 2]]
			# A[idx, 1] * (q[0] * A[idx, 1] + q[1] * A[idx, 2]) + A[idx, 2] * (q[1] * A[idx, 1] + q[2] * A[idx, 2])
			# A[idx, 1] * q[0] * A[idx, 1] + A[idx, 1] * q[1] * A[idx, 2] + A[idx, 2] * q[1] * A[idx, 1] + A[idx, 2]] * q[2] * A[idx, 2]
			# q[0] * A[idx, 1] ** 2 + 2 * A[idx, 1] * q[1] * A[idx, 2] + q[2] * A[idx, 2] ** 2 - bbar[idx] ** 2
			return numpy.asarray([
				-A[idx, 0] * A[idx, 0],
				-2 * A[idx, 0] * A[idx, 1],
				-A[idx, 1] * A[idx, 1]
			])
		return {
			'fun': val,
			'jac': jac,
			'type': 'ineq'
		}

	daConstraints = []
	for i in range(A.shape[0]):
		daConstraints.append(idx2cons(i))

	daConstraints.append({
		'fun': lambda v: v[0],
		'jac': lambda v: numpy.array([1, 0, 0]),
		'type': 'ineq'
	})
	daConstraints.append({
		'fun': lambda v: v[2],
		'jac': lambda v: numpy.array([0, 0, 1]),
		'type': 'ineq'
	})
	daConstraints.append({
		'fun': lambda v: v[0] * v[2] - v[1] * v[1],
		'jac': lambda v: numpy.asarray((v[2], -2 * v[1], v[0])),
		'type': 'ineq'
	})

	if include is not None:
		sInc = include - xbar
		daConstraints.append({
			'fun': lambda q: 2 * q[0] * q[2] * k * k - 2 * k * k * q[1] ** 2
				- q[2] * sInc[0] ** 2
				+ 2 * sInc[0] * q[1] * sInc[1]
				- q[0] * sInc[1] ** 2,
			'jac': lambda q: numpy.asarray((
				2 * q[2] * k * k - sInc[1] ** 2,
				-4 * q[1] * k * k + 2 * sInc[0] * sInc[1],
				2 * q[0] * k * k - sInc[0] ** 2
			)),
			'type': 'ineq'
		})
	result = minimize(
		lambda q: -vec2det(q),
		jac=lambda q: -vec2jac(q),
		x0=numpy.random.rand(3), constraints=daConstraints,
		method="SLSQP",
		options={"disp": False, "maxiter": 1000},
		tol=tol
	)

	if not result.success or result.fun > tol:
		return False, None

	ds = []
	lambdas = []
	for i in range(A.shape[0] - 1):  # -1 for the include constraint
		c = daConstraints[i]
		q = vec2mat(result.x) / (k * k)
		ar = A[i, :]
		br = bbar[i] / k
		qa = numpy.dot(q, ar)
		lmbda = br / numpy.dot(ar, qa)
		d = lmbda * qa
		lambdas.append(lmbda)
		ds.append(d)

	Qinv = None
	Q = None
	Linv = None
	L = None
	try:
		Qinv = vec2mat(result.x) / (k * k)
		Q = numpy.linalg.inv(Qinv)

		L = numpy.linalg.cholesky(Q).T
		Linv = numpy.linalg.inv(L)

		Qinv = Qinv / (k * k)
		Q = k * k * Q

		L = abs(k) * L
		Linv = Linv / abs(k)
	except:
		return False, None

	ellipse = Ellipse()
	ellipse.center = xbar
	ellipse.volume = result.fun
	ellipse.ds = ds
	ellipse.lambdas = lambdas
	ellipse.scale = 1.0
	ellipse.q = Q
	ellipse.q_inverse = Qinv
	ellipse.l = L
	ellipse.l_inverse = Linv
	ellipse.hot_start = None

	return True, ellipse



