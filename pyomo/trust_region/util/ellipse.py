
import numpy



class Ellipse:
	def __init__(self):
		self.center = None
		self.volume = None
		self.Q = None
		self.ds = None
		self.lambdas = None
		self.scale = 1.0

	def evaluate(self, v):
		pass

	def shift(self, v):
		pass

	def unshift(self, v):
		pass

	#'A': A,
	#'b': b,
	#'center': xbar,
	#'volume': -result.fun,
	#'Q': Q,
	#'ds': ds,
	#'lambdas': lambdas,
	#'shift': lambda v: sqrt(0.5) * dot(L, v - xbar),
	#'unshift': lambda v: xbar + sqrt(2) * dot(Linv, v),
	#'fun': lambda v: 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)),
	#'jac': lambda v: -dot(Q, v - xbar),
	#'scaled_fun': lambda scale: lambda v: 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)) / scale,
	#'scaled_jac': lambda scale: lambda v: -dot(Q, v - xbar) / scale,
	#'success': True,
	#'include': include


class EllipseParams:
	def __init__(self):
		self.center = None
		self.A = None
		self.b = None
		self.normalize = False
		self.tolerance = 1e-4
		self.include_point = None
		self.center = None
		self.include_as_constraint = False


def compute_maxamal_ellipse(p):
	k = 1.0
	#if p.normalize and p.include_point is None:
	#	k = 1 / min(abs(bbar))
	#	bbar = k * bbar

	bbar = p.b - numpy.dot(p.A, p.b.T)

	model = None
	model.constraints = ConstraintsList()

	for i in range(p.A.shape[0]):
		# [A[i, 1], A[i, 2]] * [[q[0], q[1]], [q[1], q[2]]] * [A[i, 1], A[i, 2]].T
		# [A[i, 1], A[i, 2]] * [q[0] * A[i, 1] + q[1] * A[i, 2], q[1] * A[i, 1] + q[2] * A[i, 2]].T
		# A[i, 1] * (q[0] * A[i, 1] + q[1] * A[i, 2]) + A[i, 2] * (q[1] * A[i, 1] + q[2] * A[i, 2])
		# A[i, 1] * q[0] * A[i, 1] + A[i, 1] * q[1] * A[i, 2] + A[i, 2] *q[1] * A[i, 1] + A[i, 2] * q[2] * A[i, 2]
		# A[i, 1] ** 2 * q[0] + 2 * A[i, 1] * q[1] * A[i, 2] + A[i, 2] ** 2 * q[2]

		model.constraints.add(
			p.A[i, 1] ** 2 * model.q0 +
			2 * p.A[i, 1] * model.q1 * p.A[i, 2] +
			p.A[i, 2] ** 2 * model.q2 <= bbar[i] * bbar[i] / 2
		)

	#if include is not None and include_as_constraint:
	#	sInc = include - xbar
	#	# We want:
	#	#  0.5 * include.T Q include <= 1
	#	#  include.T ( q[0] q[1] ; q[1] q[2] )^-1 include <= 2
	#	#  include.T ( q[2] -q[1] ; -q[1] q[0] ) include <= 2 * q[0] * q[2] - 2 * q[1] ** 2
	#	#  0 <= 2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] -q[1] ; -q[1] q[0] ) include
	#	#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] -q[1] ; -q[1] q[0] ) include >= 0
	#	#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] * include[0] - q[1] * include[1] ; -q[1] * include[0] + q[0] * include[1]) >= 0
	#	#  2 * q[0] * q[2] - 2 * q[1] ** 2 - (include[0] * q[2] * include[0] - include[0] * q[1] * include[1]
	#	# 			 - include[1] * q[1] * include[0] + include[1] * q[0] * include[1]) >= 0
	#	#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include[0] * q[2] * include[0] + include[0] * q[1] * include[1]
	#	# 			 + include[1] * q[1] * include[0] - include[1] * q[0] * include[1] >= 0
	#	#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include[0] * q[2] * include[0] + 2 * include[0] * q[1] * include[1]
	#	# 			 - include[1] * q[0] * include[1] >= 0
	#	#  2 * q[0] * q[2] - 2 * q[1] ** 2 - q[2] * include[0] ** 2 + 2 * include[0] * q[1] * include[1]
	#	# 			 - q[0] * include[1] ** 2 >= 0
	#	model.constraints.add(
	#		2 * model.q0 * q[2] -
	#		2 * model.q1 ** 2 -
	#		model.q2 * sInc[0] ** 2 +
	#		2 * sInc[0] * model.q1 * sInc[1] -
	#		model.q0 * sInc[1] ** 2 >= 0
	#	})

	# eigenvalue 1
	model.constraints.add(model.q0 >= 0)
	# eigenvalue 2
	model.constraints.add(model.q0 * model.q2 - model.q1 * model.q1 >= 0)

	# Solve it

	q_inverse = numpy.array([
		[model.q0(), model.q1()],
		[model.q1(), model.q2()]
	]) / (k * k)

	ds = []
	lambdas = []
	for i in range(A.shape[0]):
		ar = p.A[i, :]
		br = bbar[i] / k
		qa = numpy.dot(q_inverse, ar)
		lmbda = br / numpy.dot(ar, qa)
		d = lmbda * qa
		lambdas.append(lmbda)
		ds.append(d)

	q = inv(q_inverse)

	Linv = None
	L = None
	try:
		L = cholesky(Q)
		Linv = inv(L)

		Qinv = Qinv / (k * k)
		Q = k * k * Q

		L = abs(k) * L
		Linv = Linv / abs(k)
	except:
		return { 'success': False, 'volume': -1 }
	def get_scale(point):
		lower = 1e-12
		while scaled_fun(lower)(point) > 0:
			lower *= 2
		return lower



def getMaximalEllipse_inner(A, b, xbar, include_as_constraint, normalize=True, include=None, tol=1e-8):
	vec2mat = lambda q: asarray([[q[0], q[1]], [q[1], q[2]]])
	vec2det = lambda q: q[0] * q[2] - q[1] * q[1]
	vec2jac = lambda q: asarray([q[2], -2 * q[1], q[0]])
	daConstraints = []


	result = minimize(
		lambda q: -vec2det(q),
		jac=lambda q: -vec2jac(q),
		x0=rand(3), constraints=daConstraints,
		method="SLSQP",
		options={"disp": False, "maxiter": 1000},
		tol=tol
	)

	if not result.success or result.fun > tol:
		return {'success': False, 'volume': -1}

	ds = []
	lambdas = []
	for i in range(A.shape[0] - 1):  # -1 for the include constraint
		c = daConstraints[i]
		q = vec2mat(result.x) / (k * k)
		ar = A[i, :]
		br = bbar[i] / k
		qa = dot(q, ar)
		lmbda = br / dot(ar, qa)
		d = lmbda * qa
		lambdas.append(lmbda)
		ds.append(d)

	Qinv = None
	Q = None
	Linv = None
	L = None
	try:
		Qinv = vec2mat(result.x) / (k * k)
		Q = inv(Qinv)

		L = cholesky(Q)
		Linv = inv(L)

		Qinv = Qinv / (k * k)
		Q = k * k * Q

		L = abs(k) * L
		Linv = Linv / abs(k)
	except:
		return { 'success': False, 'volume': -1 }
	def get_scale(point):
		lower = 1e-12
		while scaled_fun(lower)(point) > 0:
			lower *= 2
		return lower

	returnVal = {
		'A': A,
		'b': b,
		'center': xbar,
		'volume': -result.fun,
		'Q': Q,
		'ds': ds,
		'lambdas': lambdas,
		'shift': lambda v: sqrt(0.5) * dot(L, v - xbar),
		'unshift': lambda v: xbar + sqrt(2) * dot(Linv, v),
		'fun': lambda v: 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)),
		'jac': lambda v: -dot(Q, v - xbar),
		'scaled_fun': lambda scale: lambda v: 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)) / scale,
		'scaled_jac': lambda scale: lambda v: -dot(Q, v - xbar) / scale,
		'success': True,
		'include': include
	}

	if not include_as_constraint:
		# 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)) / scale == 0
		# scale - 0.5 * dot(point - xbar, dot(Q, point - xbar)) == 0
		# scale == 0.5 * dot(point - xbar, dot(Q, point - xbar))
		returnVal['include_point_scale'] = max(1, 0.5 * dot(include - xbar, dot(Q, include - xbar)))
	# Sanity check for debugging
	foo = rand(2)
	foo
	el = returnVal
	# el = self.consOpts.ellipse
	foo2 = el['shift'](el['unshift'](foo))
	foo3 = el['unshift'](el['shift'](foo))
	er = max(norm(foo - foo2), norm(foo - foo3))
	if er > 0.0000001:
		raise Exception('uh oh')

	return returnVal

