

# This is currently scratch code...
# This could generate a a function and its gradient based on the basis methods...
# Would have to unite this with the test functions

from numpy import empty



class Function:
	def __init__(self):
		pass
	def copy(self):
		pass
	def evaluate(self, x):
		pass
	def getInDim(self):
		pass
	def getOutDim(self):
		pass
	def differentiate(self, var):
		pass


class Jacobian:
	def __init__(self, function):
		pass
	def evaluate(self, x):
		pass


class Gradient:
	def __init__(self, function):
		self.comps = [function.copy().differentiate(i) for i in range(0, function.getInDim())]

class Hessian:
	def __init__(self, function):
		self.comps = [[function.copy().differentiate(i).differentiate(j) for i in range(0, function.getInDim())] for j in range(0, function.getInDim())]

	def evaluate(self, x):
		n = len(self.comps)
		retVal = empty([n, n])
		for i in range(0, n):
			for j in range(0, n):
				retVal[i,j] = self.comps[i][j].evaluate(x)
		return retVal


#
# class DerivativeFreeProblem:
#
# 	def __init__(self, gamma=2):
# 		self.objective = []
# 		self.equalities = set()
# 		self.inequalities = set()
# 		self.radius = 1
# 		self.gamma = gamma
#
# 	def setObjective(self, lmbda):
# 		self.objective = lmbda
#
# 	def addEqualityConstraint(self, lmbda):
# 		self.equalities.add(lmbda)
#
# 	def addInequalityConstaint(self, lmbda):
# 		self.inequalities.add(lmbda)
#
# 	def increaseRadius(self):
# 		self.radius = self.radius * self.gamma
#
# 	def increaseRadius(self):
# 		self.radius = self.radius / self.gamma
#
# 	def ensurePoised(self):
# 		pass


class IndexGenerator:
	def __init__(self, n, k):
		self.n = n
		self.k = k
		self.current = zeros([n, 1])
		#self.current[0] = k;
	def increment(self):
		return self._increment(self.n-1)
	def _increment(self, ndx):
		if ndx < 0:
			return False
		if self.current[ndx] >= self.k:
			if not self._increment(ndx - 1):
				return False
			self.current[ndx] = 0
			return True
		self.current[ndx] += 1
		return True