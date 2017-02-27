# This code was translated from http://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/28013/versions/17/previews/RRD%20-%20ln/funct/trust.m/index.html

from numpy.linalg import eig
from numpy.linalg import norm
from numpy import zeros
from numpy import argmin
from numpy import sign
from numpy import divide
from numpy import multiply
from numpy import ones
from numpy import sqrt
from numpy import inf
from numpy import asarray
from numpy import asmatrix
from numpy import logical_and
from numpy import max
from numpy import maximum
from numpy import finfo

eps = finfo(float).eps

def seceqn(lmbda,eigval,alpha,delta):
	m = 1
	n = len(eigval)
	unn = asmatrix(ones([n, 1]))
	unm = asmatrix(ones([m, 1]))
	M = eigval * unm.T  + unn * asmatrix(lmbda).T
	MC = M.copy()
	MM = alpha * unm.T

	idx = asarray(M != 0).flatten()
	M[idx] = divide(MM[idx], M[idx])

	idx = asarray(MC == 0).flatten()
	M[idx] = inf
	M = multiply(M, M)
	# this sometimes produces division by zero
	value = sqrt(divide(unm, M.T * unn))
	value[value != value] = 0
	value = (1/delta)*unm - value
	return value[0,0]

def rfzero(x, itbnd, eigval, alpha, delta, tol=eps):
	itfun = 0
	#  if x != 0:
	#  	dx = x / 20
	# if x != 0:
	# 	dx = abs(x) / 20
	if x != 0:
		dx = abs(x) / 2
	else:
		dx = .5
	a = x
	c = a
	fa = seceqn(a, eigval, alpha, delta)
	itfun += 1
	#b = x + dx # wtf?
	b = x + 1
	fb = seceqn(b, eigval, alpha, delta)
	itfun += 1

	while (fa > 0) == (fb > 0) and itfun <= itbnd:
		dx *= 2
		# a = x - dx
		# fa = seceqn(a, eigval, alpha, delta)
		# if (fa > 0) != (fb > 0):
		# 	break
		b = x + dx
		fb = seceqn(b, eigval, alpha, delta)
		itfun += 1

	fc = fb
	while fb != 0:
		if (fb > 0) == (fc > 0):
			c = a
			fc = fa

			d = b - a
			e = d
		if abs(fc) < abs(fb):
			a = b
			b = c
			c = a
			fa = fb
			fb = fc
			fc = fa
		if itfun > itbnd:
			break
		m = .5 * (c - b)
		toler = 2 * tol * max([abs(b), 1])
		if (abs(m) <= toler) or (fb == 0):
			break
		if (abs(e) < toler) or (abs(fa) <= abs(fb)):
			# Bisection
			d = m
			e = m
		else:
			# Interpolation
			s = fb/fa
			if a == c:
				p = 2 * m * s
				q = 1 - s
			else:
				q = fa/fc
				r = fb/fc
				p = s * (2 * m * q * (q-r) - (b-a)*(r-1))
				q = (q-1)*(r-1)*(s-1)
			if p > 0:
				q = -q
			else:
				p = -p
			if 2*p < 3*m*q-abs(toler*q) and p < abs(.5*e*q):
				e = d
				d = p / q
			else:
				d = m
				e = m

		a = b
		fa = fb
		if abs(d) > toler:
			b += d
		elif b > c:
			b -= toler
		else:
			b += toler
		fb = seceqn(b, eigval, alpha, delta)
		itfun += 1
	return b, c, itfun


#function[s, val, posdef, count, lambda ] = trust(g, H, delta)
def trust(g, H, delta):
	# INITIALIZATION
	tol = 1e-12
	tol2 = 1e-8
	key = 0
	itbnd = 50
	lmbda = 0
	n = len(g)
	coeff = asmatrix(zeros(n)).T
	D, V = eig(H)
	count = 0
	eigval = asmatrix(D).T
	V = asmatrix(V)

	jmin = argmin(D)
	mineig = D[jmin]

	alpha = -V.T * g

	sig = (sign(alpha[jmin]) + (alpha[jmin] == 0))[0, 0]

	# Positive Definite Case
	if mineig > 0:
		coeff = divide(alpha, eigval)
		lmbda = 0
		s = V * coeff
		posdef = True
		nrms = norm(s)

		if nrms <= 1.2 * delta:
			key = 1
		else:
			laminit = 0
	else:
		laminit = -mineig
		posdef = False

	# Indefinite case:
	if key == 0:
		if seceqn(laminit, eigval, alpha, delta) > 0:
			b, c, count = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
			vval = abs(seceqn(b, eigval, alpha, delta))
			if vval <= tol2:
				lmbda = b
				key = 2
				lam = lmbda * asmatrix(ones([n, 1]))
				w = eigval + lam
				arg1 = logical_and(w == 0, alpha == 0)
				arg2 = logical_and(w == 0, alpha != 0)
				idx = asarray(w != 0).flatten()
				coeff[idx] = divide(alpha[idx], w[idx])
				coeff[arg1] = 0
				coeff[arg2] = inf
				coeff[coeff != coeff] = 0
				s = V * coeff
				nrms = norm(s)
				if nrms > 1.2 * delta or nrms < .8 * delta:
					key = 5
					lmbda = -mineig
			else:
				lmbda = -mineig
				key = 3
		else:
			lmbda = -mineig
			key = 4
		lam = lmbda * asmatrix(ones([n, 1]))
		if key > 2:
			arg = abs(eigval + lam) < (10 * eps) * maximum(abs(eigval), asmatrix(ones(eigval.shape)))
			alpha[arg] = 0
		w = eigval + lam
		arg1 = logical_and(w == 0, alpha == 0)
		arg2 = logical_and(w == 0, alpha != 0)
		idx = asarray(w != 0).flatten()
		coeff[idx] = divide(alpha[idx], w[idx])
		coeff[arg1] = 0
		coeff[arg2] = inf
		coeff[coeff != coeff] = 0
		s = V * coeff
		nrms = norm(s)
		if key > 2 and nrms < .8 * delta:
			beta = sqrt(delta ** 2 - nrms ** 2)
			s += beta * sig * V[:, jmin]
		if key > 2 and nrms > 1.2 * delta:
			b, c, count = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
			lmbda = b
			lam = lmbda * asmatrix(ones([n, 1]))
			w = eigval + lam
			arg1 = logical_and(w == 0, alpha == 0)
			arg2 = logical_and(w == 0, alpha != 0)
			idx = asarray(w != 0).flatten()
			coeff[idx] = divide(alpha[idx], w[idx])
			coeff[arg1] = 0
			coeff[arg2] = inf
			coeff[coeff != coeff] = 0
			s = V * coeff
			nrms = norm(s)
	val = g.T * s + .5 * s.T * H * s
	return s, val[0,0], posdef, count, lmbda
