from math import cos
from math import sin

from numpy import array as arr
from numpy import dot
from numpy import zeros
from numpy.linalg import norm as norm
from numpy import empty
from numpy import sqrt
from numpy import asarray
from numpy import copy

from numpy import matrix

# objective functions have:
#	evaluate
#	getInDim
#	gradient
#	hessian

# constraint functions have:
#	evaluate
#	getInDim
#	gradient

# vector functions have:
#	evaluate
#	getInDim
#	jacobian


# vector function

class EmptyFunction:
	def __init__(self):
		pass
	def evaluate(self, x):
		return None
	def gradient(self, x):
		return empty(0)
	def hessian(self, x):
		return empty([0, 0])

class VectorFunction:
	def __init__(self, n):
		self.funs = []
		self.n = n

	def add(self, fun):
		self.funs.append(fun)

	def evaluate(self, x):
		retVal = empty((self.getOutDim()))
		for i in range(0, len(retVal)):
			retVal[i] = self.funs[i].evaluate(x)
		return retVal

	def getFunction(self, i):
		return self.funs[i]

	def jacobian(self, x):
		retVal = empty((self.getOutDim(), self.n))
		for i in range(0, len(retVal)):
			retVal[i, :] = self.funs[i].gradient(x)
		return retVal

	def getOutDim(self):
		return len(self.funs)

class VectorValuedQuadratic:
	def __init__(self, n, m):
		self.Q = empty([m, n, n])
		self.b = empty([m, n])
		self.c = empty(m)

	def set(self, idx, fun):
		self.Q[idx, :, :] = fun.Q[:, :]
		self.b[idx, :] = fun.b[:]
		self.c[idx] = fun.c

	def evaluate(self, x):
		return dot(dot(self.Q, x), .5 * x) + dot(self.b, x) + self.c

	def jacobian(self, x):
		return dot(self.Q, x) + self.b

	def hessian(self, x):
		return self.Q

	def getOutDim(self):
		return self.Q.shape[0]

	def getInDim(self):
		return self.Q.shape[1]


# Objective functions:
class Quadratic:
	def __init__(self, Q, b, c):
		self.Q = asarray(Q)
		self.b = asarray(b)
		if isinstance(c, matrix):
			self.c = c[0,0]
		else:
			self.c = c

	def evaluate(self, x):
		return .5 * dot(x, dot(self.Q, x)) + dot(self.b, x) + self.c

	def gradient(self, x):
		return dot(self.Q, x) + self.b
	def hessian(self, x):
		return copy(self.Q)
	def getInDim(self):
		return self.Q.shape[0]
	def getOutDim(self):
		return 1


	# ought to be cleaned up... (and deleted.)
	def hitsZero(self, direction, tol):
		a = .5 * dot(direction, dot(self.Q, direction))
		b = dot(self.b, direction)
		c = self.c

		disc = b*b - 4*a*c
		if disc < 0:
			raise Exception("uh oh!!!!")

		den = 2*a
		root = (-b - sqrt(disc)) / den
		if root < 0:
			root = (-b + sqrt(disc)) / den

		if root < 0:
			raise Exception("uh oh!!!!")

		return root


	def shift(self, center, radius):
		newQ = self.Q / radius / radius
		newB = (dot(self.Q, center)/radius + self.b) / radius
		newC = self.c + dot(self.b, center) / radius + .5 * dot(center, dot(self.Q, center)) / radius / radius
		return Quadratic(newQ, newB, newC)

class Line:
	def __init__(self, vec, val):
		self.vec = vec
		self.val = val
	def evaluate(self, x):
		return dot(self.vec, x) + self.val
	def gradient(self, x):
		return self.vec
	def hessian(self, x):
		return zeros((self.getInDim(), self.getInDim()))
	def getInDim(self):
		return len(self.vec)
	def getOutDim(self):
		return 1
	def shift(self, center, radius):
		newVec = self.vec / radius
		newVal = self.val + dot(self.vec, center) / radius
		return Line(newVec, newVal)

class Rosenbrock:
	def __init__(self, a, b):
		self.a = a
		self.b = b

	def evaluate(self, x):
		return (self.a - x[0])**2 + self.b*(x[1] - x[0]**2)**2

	def gradient(self, x):
		return arr([
			-2 * (self.a - x[0]) + 2 * self.b * (x[1] - x[0] ** 2) * (-2 * x[0]),
			2 * self.b * (x[1] - x[0] ** 2)
		])
	def hessian(self, x):
		return arr([[
			2 * x[0] - 4 * self.b * (x[1] - 3 * x[0] ** 2),
			2 * self.b * (x[1] - x[0] ** 2)
		], [
			2 * self.b * (-2 * x[0]),
			2 * self.b * x[1]
		]])
	def getInDim(self):
		return 2
	def getOutDim(self):
		return 1

# Constraint Functions
class DistanceToCircle:
	def __init__(self, center, r):
		self.center = center
		self.r = r
	def evaluate(self, x):
		return (norm(x - self.center) - self.r)**2
	def gradient(self, x):
		d = norm(x-self.center)
		return 2*(d-self.r)/d*(x-self.center)
	def hessian(self, x):
		raise Exception('not implemented')
	def getInDim(self):
		return self.center.shape[0]
	def getOutDim(self):
		return 1

class TrustRegionConstraint:
	def __init__(self, center, r):
		self.center = center
		self.r = r

	def evaluate(self, x):
		return max(norm(x - self.center) - self.r, 0)

	def gradient(self, x):
		return (2*(d-self.r)/d)*(x-self.center)
	def hessian(self, x):
		raise Exception('not implemented')
	def getInDim(self):
		return len(self.center)
	def getOutDim(self):
		return 1

class DistanceToInsideCircle:
	def __init__(self, center, r):
		self.center = center
		self.r = r
	def evaluate(self, x):
		return max(norm(x - self.center) - self.r, 0)
	def gradient(self, x):
		d = norm(x-self.center)
		if d <= self.r:
			return zeros(x.shape)
		else:
			return (2*(d-self.r)/d)*(x-self.center)
	def hessian(self, x):
		raise Exception('not implemented')
	def getInDim(self):
		return len(self.center)
	def getOutDim(self):
		return 1


class DistanceToLine:
	def __init__(self, vec, val):
		self.vec = vec
		self.val = val
	def evaluate(self, x):
		return (dot(self.vec, x) - self.val)**2
	def gradient(self, x):
		return 2 * (dot(self.vec, x) - self.val) * self.vec
	def hessian(self, x):
		raise Exception('not implemented')
	def getInDim(self):
		return len(self.vec)
	def getOutDim(self):
		return 1

class DistanceToSin:
	def __init__(self, a=5):
		self.a = a
	def evaluate(self, x):
		return (x[0] - self.a * sin(x[1]))**2
	def gradient(self, x):
		return arr([
			2 * (x[0] - self.a * sin(x[1])),
			2 * (x[0] - self.a * sin(x[1])) * (-self.a * cos(x[1]))
		])
	def hessian(self, x):
		raise Exception('not implemented')
	def getInDim(self):
		return 2
	def getOutDim(self):
		return 1

class Wiggles:
	def __init__(self):
		pass
	def evaluate(self, x):
		return x[0] - sin(x[1])
	def gradient(self, x):
		return arr([
			1,
			-cos(x[1])
		])
	def hessian(self, x):
		return arr([[
			1, 0
		], [
			0, sin(x[1])
		]])
	def getInDim(self):
		return 2
	def getOutDim(self):
		return 1






