# from numpy import array
# from numpy import empty
from numpy.linalg import cholesky
from numpy.random import rand
from numpy.linalg import norm

from numpy.linalg import eig
from numpy import copy
from numpy import linspace
from numpy import asarray
from numpy import array
from numpy import empty
from numpy import sum
from numpy import multiply
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
import numpy as np
import time

import operator




#
def e(i):
	ret = zeros(3)
	ret[i] = 1
	return ret

def getMaximalEllipse_inner(A, b, xbar, normalize=True, include=None, tol=1e-8):
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
		options={ "disp": False, "maxiter": 1000 },
		tol=tol
	)

	if not result.success or result.fun > tol:
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


def getMaximalEllipse(A, b, xbar):
	# ellipse1 = getMaximalEllipse_inner(A, b, xbar, normalize=False)
	ellipse2 = getMaximalEllipse_inner(A, b, xbar, normalize=True, include=None)
	return ellipse2

def getSearchDirection(xbar, A, b):
	distances = abs(dot(A, xbar) - b) / sum(multiply(A, A), axis=1)
	min_index, _ = min(enumerate(distances), key=operator.itemgetter(1))
	return A[min_index, :]






class SearchPath:
	def __init__(self, points):
		self.points = points

	def get_point(self, t):
		if len(self.points) == 1:
			return self.points[0]

		idx = min(
			int(t * (len(self.points) - 1)),
			len(self.points) - 2
		)
		l1 = (idx + 0) / (len(self.points) - 1)
		l2 = (idx + 1) / (len(self.points) - 1)
		between = (l2 - l1)

		return (l2 - t) / between * self.points[idx] + (t - l1) / between * self.points[idx + 1]

	def plot(self, ax):
		for i in range(len(self.points) - 1):
			p1 = self.points[i]
			p2 = self.points[i+1]

			ax.add_patch(patches.Arrow(
				x=p1[0], y=p1[1],
				dx=p2[0] - p1[0], dy=p2[1] - p1[1],
				facecolor="red",
				edgecolor="red",
				width=0.05
			))





def get_search_path(x, A, b):
	all_points = []
	all_points.append(x)

	A = copy(A).astype(float)
	b = copy(b).astype(float)
	for i in range(A.shape[0]):
		n = norm(A[i, :])
		A[i, :] /= n
		b[i] /= n

	distances = abs(dot(A, x) - b) / sum(multiply(A, A), axis=1)
	s_distances = sorted(enumerate(distances), key=operator.itemgetter(1))
	idx_min = s_distances[0][0]
	idx_sec = s_distances[1][0]

	if abs(s_distances[0][1] - s_distances[1][1]) < 1e-12:
		return SearchPath(all_points)

	Ac = A[idx_min, :]
	Ab = A[idx_sec, :]
	first_direction = -Ac * -1
	second_direction = -(Ac + Ab) / 2 * -1
	if norm(second_direction) < 1e-12:
		# BANG HEAD AGAINST TABLE
		Ab = A[s_distances[2][0]]
		second_direction = -(Ac + Ab) / 2
		# SOMEHOW NEED TO MAKE SURE THEY ARE LINEARLY INDEPENDENT

	dd = -dot(A, first_direction)
	t_min = None
	for i in range(A.shape[0]):
		if abs(dd[idx_min] - dd[i]) < 1e-12:
			continue
		t_intersection = (distances[i] - distances[idx_min]) / (dd[idx_min] - dd[i]) * -1
		if t_intersection < 0:
			print("oh boy")
		if t_min is None or t_intersection < t_min:
			t_min = t_intersection

	p1 = x + t_min * first_direction
	all_points.append(p1)

	d2 = abs(dot(A, p1) - b) / sum(multiply(A, A), axis=1)
	print("After first point", d2)

	dd = -dot(A, second_direction)
	t_min = None
	for i in range(A.shape[0]):
		if abs(dd[idx_min] - dd[i]) < 1e-12:
			continue
		t_intersection = (d2[i] - d2[idx_min]) / (dd[idx_min] - dd[i]) * -1
		if t_min is None or t_intersection < t_min:
			t_min = t_intersection

	p2 = p1 + t_min * second_direction
	all_points.append(p2)

	d3 = abs(dot(A, p2) - b) / sum(multiply(A, A), axis=1)
	print("After second point", d3)

	return SearchPath(all_points)


		# :,(             !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#t_crit = 0
	#radius = 100
	#center = radius / 2
	#while radius > 1e-12:
	#	trials = linspace(center - radius, center + radius, 1000)
	#	all_distances = [(trial, abs(
	#		abs(dot(Ac, start - trial * Ac)) / (Ac[0] * Ac[0] + Ac[1] * Ac[1]) -
	#		abs(dot(Ab, start - trial * Ac)) / (Ab[0] * Ab[0] + Ab[1] * Ab[1])
	#	)) for trial in trials]
	#	min_index, _ = min(enumerate(all_distances), key=operator.itemgetter(1))
	#	center = all_distances[min_index][0]
	#	radius /= 2
	#
	# intersection = start - t_crit * Ac
	# #pivot_distance = norm(t_crit * Ac)
	#
	# def search_path(t):
	# 	if t < t_crit:
	# 		return start + t * first_direction
	# 	else:
	# 		return intersection + (t - t_crit) * second_direction
	#
	# return search_path





def getMaximalEllipseContaining(A, b, xbar, tol=1e-8):
	maxCenter = xbar
	maxEllipse = getMaximalEllipse_inner(
		A=A,
		b=b,
		xbar=xbar,
		normalize=False,
		include=xbar
	)
	if maxEllipse['success']:
		plotEllipse(maxEllipse)
	else:
		return

	#while False and not (dot(A, xbar) + tol >= b).all():
	#	feasibility = dot(A, xbar) - b
	#	direction = None
	#	for i in range(len(feasibility)):
	#		if feasibility[i] < -tol:
	#			direction = A[i, :] / norm(A[i, :]) * abs(feasibility[i]) / sqrt(A[i, 0] ** 2 + A[i, 1] ** 2)
	#			break
	#	if direction is None:
	#		raise Exception('uh oh')
	#	for _ in range(1000):
	#		delta = 1 + 0.5 * rand()
	#		if not (dot(A, xbar + delta * direction) >= b).all():
	#			continue
	#
	#		# worst hack of all time
	#		xbar = xbar + delta * direction
	#		maxCenter = xbar
	#		maxEllipse = getMaximalEllipse_inner(
	#			A=A,
	#			b=b,
	#			xbar=xbar,
	#			normalize=False,
	#			include=xbar,
	#			tol=tol
	#		)
	#		maxEllipse['include'] = xbar
	#		if not maxEllipse['success']:
	#			continue
	#		plotEllipse(maxEllipse)
	#		break
	#nd = getSearchDirection(xbar, A, b)
	#nd = nd / norm(nd)

	search_path = get_search_path(xbar, A, b)

	# EXTREMELY dumb search!!!!!!!!!!!
	delta = 0.5
	center = 0.5
	TRIAL_POINTS = 10
	while delta > 1e-12 and len(search_path.points) > 1:
		improved = False

		for t in linspace(center - delta, center + delta, TRIAL_POINTS):
			otherCenter = search_path.get_point(t)
			if not (dot(A, otherCenter) >= b).all():
				continue

			otherEllipse = getMaximalEllipse_inner(
				A=A,
				b=b,
				xbar=otherCenter,
				normalize=False,
				include=xbar
			)

			if not otherEllipse['success']:
				continue

			plotEllipse(otherEllipse)

			if otherEllipse['volume'] <= maxEllipse['volume']:
				continue

			# plotEllipse(maxEllipse)

			maxEllipse = otherEllipse
			maxCenter = otherCenter
			improved = True
			center = t

		delta = min(
			2 * delta / TRIAL_POINTS,
			center,
			1 - center
		)
	maxEllipse['center'] = maxCenter
	if maxEllipse['success']:
		plotEllipse(maxEllipse)
	return maxEllipse


def plotEllipse_inner(ellipse, ax, bounds, scaled=None):
	mat2cons = lambda idx: lambda x: dot(A[idx, :], x) - b[idx]
	ds = ellipse['ds']
	A = ellipse['A']
	b = ellipse['b']
	x = ellipse['center']
	include = ellipse['include']

	# t = linspace(0, 2 * M, 1000)
	ax.plot(array(x[0]), array(x[1]), 'bo')
	if include is not None:
		ax.plot(array(include[0]), array(include[1]), 'ro')

	tx = linspace(bounds['lbX'], bounds['ubX'], num=100)
	ty = linspace(bounds['lbY'], bounds['ubY'], num=100)
	X, Y = meshgrid(tx, ty)
	Z = empty((len(ty), len(tx)))

	def plotContour(fun, color, levels=[-1, 0]):
		for i in range(0, len(tx)):
			for j in range(0, len(ty)):
				Z[j, i] = fun(array([tx[i], ty[j]]))
		CS = plt.contour(X, Y, Z, levels, colors=color)

	for i in arange(A.shape[0]):
		plotContour(mat2cons(i), 'b')

	plotArrows = 'plotArrows' not in bounds or bounds['plotArrows']
	if plotArrows:
		for d in ds:
			ax.add_patch(patches.Arrow(
				x=x[0], y=x[1],
				dx=d[0], dy=d[1],
				facecolor="green",
				edgecolor="green",
				width=0.05
			))

		nd = getSearchDirection(x, A, b)
		if include is not None:
			nd = getSearchDirection(include, A, b)
		normalizing = ( bounds['ubX'] - bounds['lbX'] ) / (4 * norm(nd))
		ax.add_patch(patches.Arrow(
			x=x[0], y=x[1],
			dx=nd[0] * normalizing, dy=nd[1] * normalizing,
			facecolor="pink",
			edgecolor="pink",
			width=0.05
		))
		ax.add_patch(patches.Arrow(
			x=x[0], y=x[1],
			dx=-nd[0] * normalizing, dy=-nd[1] * normalizing,
			facecolor="pink",
			edgecolor="pink",
			width=0.05
		))

	search_path = get_search_path(x, A, b)
	search_path.plot(ax)

	# should be include if it is there...
	#search_path = get_search_path(start=x, A=A, b=b)
	#path_points = [search_path(t) for t in linspace(0, 5, 100)]
	#ax.plot(
	#	array([point[0] for point in path_points]),
	#	array([point[1] for point in path_points]),
	#	'mx'
	#)

	if 'fun' in ellipse:
		plotContour(ellipse['fun'], 'k', levels=[0])
		if scaled is not None:
			plotContour(ellipse['scaled_fun'](scaled), 'r', levels=[0])


ellipseCount = 0
def plotEllipse(ellipse, bounds = {'lbX': -10, 'ubX': 10, 'lbY': -10, 'ubY': 10, 'plotArrows': True}):
#def plotEllipse(ellipse, bounds = {'lbX': -10, 'ubX': 10, 'lbY': -10, 'ubY': 10}):
	fig = plt.figure()
	ax = plt.axes()
	ax.set_xlim([bounds['lbX'], bounds['ubX']])
	ax.set_ylim([bounds['lbY'], bounds['ubY']])
	# plt.show()

	plotEllipse_inner(ellipse, bounds=bounds, ax=ax)

	global ellipseCount
	ellipseCount += 1
	plt.savefig('images/ellipse_' + str(ellipseCount) + '.png')
	# plt.show()
	plt.close()
# f = lambda x: 0.5 * dot(x, dot(Q, x))
# fg = lambda x: dot(Q, x)
# for i in range(len(ds)):
# 	print('gradient of ellipse', fg(ds[i]))
# 	print('gradient of constraint', lambdas[i] * A[i, :])
#
# print('all done')


