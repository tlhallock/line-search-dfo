
from math import factorial


from scipy.misc   import comb
from numpy        import int       as integral
from numpy        import multiply
from numpy import empty
from numpy import power
from numpy import prod
from numpy.matlib import repmat
from copy import deepcopy
from sys import stdout

from utilities import functions


# Could convert this into yield
def _fillAllPowers(current, index, balls, idxToDest, destination):
	if balls < 0 or index >= len(current):
		return idxToDest
	if index == len(current)-1:
		current[index] = balls
		destination[:, idxToDest] = current
		return idxToDest+1
	else:
		for rem in range(0, balls+1):
			current[index] = rem
			idxToDest = _fillAllPowers(current, index+1, balls - rem, idxToDest, destination)
		return idxToDest


def _getBasisDimension(n, degree):
	count = 0
	for i in range(0, degree+1):
		count += comb(i+n-1,n-1)
	return integral(count)


class PolynomialBasis:
	def __init__(self, n, degree):
		self.n = n
		self.degree = degree
		self.basis_dimension = _getBasisDimension(n, degree)

		self.powers = empty([n, self.basis_dimension], dtype=integral)
		self.coeff = empty(self.basis_dimension)

		index = integral(0)
		powers = empty(n, dtype=integral)
		for k in range(0, degree+1):
			num = integral(comb(k+n-1, n-1))
			self.coeff[index:index+num] = 1 / factorial(k)
			index = _fillAllPowers(powers, 0, k, index, self.powers)

	def evaluateMatToMat(self, mat):
		npoints = mat.shape[0]
		retVal = empty((npoints, self.basis_dimension))
		for i in range(0, npoints):
			retVal[i,:] = self.evaluateRowToRow(mat[i])
		return retVal

	def evaluateRowToRow(self, row):
		return multiply(prod(power(repmat(row, self.basis_dimension, 1).T, self.powers), axis=0), self.coeff)

	def evaluateToNumber(self, coefficients, x):
		return sum(multiply(self.evaluateRowToRow(x), coefficients))

	def evaluateAtMatrix(self, coefficients, mat):
		nrows = mat.shape[0]
		retVal = empty(nrows)
		for i in range(0, nrows):
			retVal[i] = self.evaluate(coefficients, mat[i])
		return retVal

	def print(self):
		for i in range(0, len(self.coeff)):
			stdout.write(str(round(self.coeff[i], 2)))
			stdout.write('\t')
		stdout.write('\n')
		stdout.write('\n')
		for i in range(0, self.powers.shape[0]):
			for j in range(0, self.powers.shape[1]):
				stdout.write(str(self.powers[i, j]))
				stdout.write('\t')
			stdout.write('\n')
		stdout.flush()

	def copy(self):
		return deepcopy(self)

	def differentiate(self, var):
		for i in range(0, self.powers.shape[1]):
			if self.powers[var, i] == 0:
				self.coeff[i] = 0
			else:
				self.coeff[i] = self.coeff[i] * self.powers[var, i]
				self.powers[var, i] -= 1

	def getLinearModel(self, phi):
		c=0
		b=empty(self.n)
		for j in range(self.powers.shape[1]):
			sumpowers = sum(self.powers[:, j])
			if sumpowers == 0:
				c = phi[j] * self.powers[j]
			elif sumpowers == 1:
				b[self.powers[:, j] > 0] = phi[j] * self.coeff[j]
			else:
				pass
		return functions.ALine(b, -c)



	def getQuadraticModel(self, phi):
		c=0
		b=empty(self.n)
		Q=empty((self.n, self.n))
		for j in range(self.powers.shape[1]):
			sumpowers = sum(self.powers[:, j])
			if sumpowers == 0:
				c = phi[j] * self.coeff[j]
			elif sumpowers == 1:
				b[self.powers[:, j] > 0] = phi[j] * self.coeff[j]
			elif sumpowers == 2:
				idx1=-1
				idx2=-1
				for i in range(self.powers.shape[0]):
					if self.powers[i,j] == 0:
						continue
					if idx1 < 0:
						idx1 = i
					else:
						idx2 = i
				if idx2 < 0:
					Q[idx1, idx1] = 2 * phi[j] * self.coeff[j]
				else:
					Q[idx1, idx2] = phi[j] * self.coeff[j]
					Q[idx2, idx1] = phi[j] * self.coeff[j]
			else:
				pass
		return functions.Quadratic(Q, b, c)



