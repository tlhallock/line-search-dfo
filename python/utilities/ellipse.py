# from numpy import array
# from numpy import empty
from numpy.linalg import cholesky
from numpy.linalg import eig
from numpy import linspace
from numpy import asarray
from numpy import array
from numpy import empty
from numpy.random import rand
from numpy import dot
from numpy import zeros
from numpy import arange
from numpy import meshgrid
from numpy import sqrt
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.patches as patches
import time



#
def e(i):
	ret = zeros(3)
	ret[i] = 1
	return ret

def getMaximalEllipse_inner(A, b, xbar, normalize=True, include=None, plot=True):
	k = 1
	bbar = b - dot(A, xbar.T)
	if normalize and include is None:
		k = 1 / min(abs(bbar))
		bbar = k * bbar
	vec2mat = lambda q: asarray([[q[0], q[1]], [q[1], q[2]]])
	vec2det = lambda q: q[0] * q[2] - q[1] * q[1]
	vec2jac = lambda q: asarray([q[2], -2 * q[1], q[0]])
	def idx2cons(idx):
		Ar = A[idx,:]
		br = bbar[idx]
		def val(x):
			q = vec2mat(x)
			return -(dot(Ar, dot(q, Ar)) - br * br / 2)
		def jac(q):
			# [A[idx, 1], A[idx, 2]] * [[q[0], q[1]], [q[1], q[2]]] * [A[idx, 1], A[idx, 2]]
			# [A[idx, 1], A[idx, 2]] * [q[0] * A[idx, 1] + q[1] * A[idx, 2], q[1] * A[idx, 1] + q[2] * A[idx, 2]]
			# A[idx, 1] * (q[0] * A[idx, 1] + q[1] * A[idx, 2]) + A[idx, 2] * (q[1] * A[idx, 1] + q[2] * A[idx, 2])
			# A[idx, 1] * q[0] * A[idx, 1] + A[idx, 1] * q[1] * A[idx, 2] + A[idx, 2] * q[1] * A[idx, 1] + A[idx, 2]] * q[2] * A[idx, 2]
			# q[0] * A[idx, 1] ** 2 + 2 * A[idx, 1] * q[1] * A[idx, 2] + q[2] * A[idx, 2] ** 2 - bbar[idx] ** 2
			return asarray([
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
	# eigenvalue 1
	daConstraints.append({
		'fun': lambda v: v[0],
		'jac': lambda v: e(0),
		'type': 'ineq'
	})
	# eigenvalue 2
	daConstraints.append({
		'fun': lambda v: v[0] * v[2] - v[1] * v[1],
		'jac': lambda v: asarray((v[2], -2 * v[1], v[0])),
		'type': 'ineq'
	})

	if include is not None:
		sInc = include - xbar
		# We want:
		#  0.5 * include.T Q include <= 1
		#  include.T ( q[0] q[1] ; q[1] q[2] )^-1 include <= 2
		#  include.T ( q[2] -q[1] ; -q[1] q[0] ) include <= 2 * q[0] * q[2] - 2 * q[1] ** 2
		#  0 <= 2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] -q[1] ; -q[1] q[0] ) include
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] -q[1] ; -q[1] q[0] ) include >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include.T ( q[2] * include[0] - q[1] * include[1] ; -q[1] * include[0] + q[0] * include[1]) >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - (include[0] * q[2] * include[0] - include[0] * q[1] * include[1]
		# 			 - include[1] * q[1] * include[0] + include[1] * q[0] * include[1]) >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include[0] * q[2] * include[0] + include[0] * q[1] * include[1]
		# 			 + include[1] * q[1] * include[0] - include[1] * q[0] * include[1] >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - include[0] * q[2] * include[0] + 2 * include[0] * q[1] * include[1]
		# 			 - include[1] * q[0] * include[1] >= 0
		#  2 * q[0] * q[2] - 2 * q[1] ** 2 - q[2] * include[0] ** 2 + 2 * include[0] * q[1] * include[1]
		# 			 - q[0] * include[1] ** 2 >= 0
		daConstraints.append({
			'fun': lambda q: 2 * q[0] * q[2] - 2 * q[1] ** 2
				- q[2] * sInc[0] ** 2
				+ 2 * sInc[0] * q[1] * sInc[1]
				- q[0] * sInc[1] ** 2,
			'jac': lambda q: asarray((
				2 * q[2] - sInc[1] ** 2,
				-4 * q[1] + 2 * sInc[0] * sInc[1],
				2 * q[0] - sInc[0] ** 2
			)),
			'type': 'ineq'
		})
	# result2 = divideRect.constrained(divideRect.Program({
	# 	'fun': vec2det
	# }, daConstraints, 1e-8), asarray([0, 0, 0]), asarray([20, 20, 20]))
	result = minimize(
		lambda q: -vec2det(q),
		jac=lambda q: -vec2jac(q),
		x0=rand(3), constraints=daConstraints,
		method="SLSQP",
		options={ "disp": False, "maxiter": 1000 }
	)

	if not result.success or result.fun > 0:
		return { 'success': False, 'volume': -1 }

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
	Qinv = vec2mat(result.x) / (k * k)
	Q = inv(Qinv)
	if plot:
		plotEllipse(-2, 2, -2, 2, A, b, xbar, {
			'fun': lambda v: 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar))
		}, ds, include)
#	print('found the Q and Qinv of', Q, Qinv)
	Linv = None
	L = None
	try:
		Linv = cholesky(Qinv)  # don't need to perform 2 Cholesky's here
		L = cholesky(Q)
	except:
		return { 'success': False, 'volume': -1 }

	return {
		'volume': -result.fun,
		'Q': Q,
		'ds': ds,
		'lambdas': lambdas,
		'shift': lambda v: sqrt(0.5) * dot(L, v - xbar),
		'unshift': lambda v: xbar + sqrt(2) * dot(Linv, v),
		'fun': lambda v: 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)),
		'jac': lambda v: -dot(Q, v - xbar),
		'scaled_fun': lambda scale: lambda v: 1 - 0.5 * dot(v - xbar, dot(Q, v - xbar)),
		'scaled_jac': lambda scale: lambda v: -dot(Q, v - xbar),
		'success': True
	}


def getMaximalEllipse(A, b, xbar):
	# ellipse1 = getMaximalEllipse_inner(A, b, xbar, normalize=False)
	ellipse2 = getMaximalEllipse_inner(A, b, xbar, normalize=True, include=None)
	return ellipse2

def getMaximalEllipseContaining(A, b, xbar):
	maxCenter = xbar
	maxEllipse = getMaximalEllipse_inner(
		A=A,
		b=b,
		xbar=xbar,
		normalize=False,
		include=xbar,
		plot=True
	)
	if maxEllipse['success']:
		plotEllipse(-2, 2, -2, 2, A, b, maxCenter, maxEllipse, maxEllipse['ds'], xbar)

	# EXTREMELY dumb search!!!!!!!!!!!
	delta = 1
	while delta > 1e-12:
		improved = False

		for _ in arange(100):
			direction = 2 * rand(2) - 1
			direction = delta * direction / norm(direction)

			otherCenter = maxCenter + direction
			if not (dot(A, otherCenter) <= b).all():
				continue

			otherEllipse = getMaximalEllipse_inner(
				A=A,
				b=b,
				xbar=otherCenter,
				normalize=False,
				include=xbar,
				plot=False
			)
			if not otherEllipse['success']:
				continue
			if otherEllipse['volume'] <= maxEllipse['volume']:
				continue

			maxEllipse = otherEllipse
			maxCenter = otherCenter
			improved = True

		if not improved:
			delta /= 2
	maxEllipse['center'] = maxCenter
	if maxEllipse['success']:
		plotEllipse(-2, 2, -2, 2, A, b, maxCenter, maxEllipse, maxEllipse['ds'], xbar)
	return maxEllipse




ellipseCount = 0
def plotEllipse(lbX, ubX, lbY, ubY, A, b, x, ellipse, ds, include=None):
	mat2cons = lambda idx: lambda x: dot(A[idx, :], x) - b[idx]

	fig = plt.figure()
	ax = plt.axes()
	ax.set_xlim([lbX, ubX])
	ax.set_ylim([lbY, ubY])
	# t = linspace(0, 2 * M, 1000)
	ax.plot(array(x[0]), array(x[1]), 'bo')
	if include is not None:
		ax.plot(array(include[0]), array(include[1]), 'ro')

	tx = linspace(lbX, ubX, num=100)
	ty = linspace(lbY, ubY, num=100)
	X, Y = meshgrid(tx, ty)
	Z = empty((len(ty), len(tx)))

	def plotContour(fun, color, levels=[-1, 0]):
		for i in range(0, len(tx)):
			for j in range(0, len(ty)):
				Z[j, i] = fun(array([tx[i], ty[j]]))
		CS = plt.contour(X, Y, Z, levels, colors=color)

	for i in arange(A.shape[0]):
		plotContour(mat2cons(i), 'b')

	for d in ds:
		ax.add_patch(patches.Arrow(
			x=x[0], y=x[1],
			dx=d[0], dy=d[1],
			facecolor="green",
			edgecolor="green",
			width=0.05
		))

	plotContour(ellipse['fun'], 'k')

	global ellipseCount
	ellipseCount += 1
	plt.savefig('images/ellipse_' + str(ellipseCount) + '.png')
	plt.close()
	# plt.show()

# f = lambda x: 0.5 * dot(x, dot(Q, x))
# fg = lambda x: dot(Q, x)
# for i in range(len(ds)):
# 	print('gradient of ellipse', fg(ds[i]))
# 	print('gradient of constraint', lambdas[i] * A[i, :])
#
# print('all done')


