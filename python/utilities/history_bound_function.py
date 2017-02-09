

from numpy.linalg import norm
from numpy import inf as infinity
from numpy import maximum
from numpy import minimum
from numpy import full
from numpy.random import random
from numpy import multiply
from numpy import zeros
from utilities import functions


class Evaluation:
	def __init__(self, x, Q, b, c):
		self.x = x
		self.Q = Q
		self.b = b
		self.c = c

	def updateQBounds(self, x, QL, QU, L):
		dist = norm(self.x - x)
		QL[:] = maximum(QL, self.Q - dist * L)
		QU[:] = minimum(QU, self.Q + dist * L)

	def updateBBounds(self, x, bL, bU, L):
		dist = norm(self.x - x)
		dist *= dist
		bL[:] = maximum(bL, self.b - dist * L)
		bU[:] = minimum(bU, self.b + dist * L)

	def updateCBounds(self, x, cL, cU, L):
		dist = norm(self.x - x)
		dist *= dist * dist
		cL = maximum(cL, self.c - dist * L)
		cU = minimum(cU, self.c + dist * L)
		return cL, cU

class HistoryBoundRandomFunction:
	def __init__(self, maxChangeInParams, n):
		self.n = n
		self.history = [Evaluation(zeros(n), random([self.n, self.n]), random(self.n), random())]
		self.maxChangeInParams = maxChangeInParams

	def evaluate(self, x):
		qL = full([self.n, self.n], -infinity)
		qU = full([self.n, self.n],  infinity)
		bL = full(self.n, -infinity)
		bU = full(self.n,  infinity)
		cL = -infinity
		cU =  infinity

		for e in self.history:
			e.updateQBounds(x, qL, qU, self.maxChangeInParams)
			e.updateBBounds(x, bL, bU, self.maxChangeInParams)
			cL, cU = e.updateCBounds(x, cL, cU, self.maxChangeInParams)

		rand = random([self.n, self.n])
		newQ = multiply(rand, qL) + multiply(1-rand, qU)
		rand = random(self.n)
		newB = multiply(rand, bL) + multiply(1-rand, bU)
		rand = random()
		newC = rand * cL + (1-rand) * cU

		self.history.append(Evaluation(x, newQ, newB, newC))
		return functions.Quadratic(2 * newQ, newB, newC)
