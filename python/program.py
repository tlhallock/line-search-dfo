
from numpy import empty
from numpy.linalg import norm
from dfo import polynomial_basis
from dfo import dfo_model
from numpy import array as arr
import matplotlib.pyplot as plt

from numpy import linspace
from numpy import meshgrid
from utilities import sys_utils

import dfo
import matplotlib

class Program:
	def __init__(self, name, obj, eq, ineq, x0, max_iters):
		self.eq = eq
		self.ineq = ineq
		self.name = name
		self.f = obj
		self.h = eq
		self.g = ineq
		self.x0 = x0
		self.tol = 1e-8
		self.imageNumber = 0
		self.max_iters = max_iters

	def getImageNumber(self):
		returnValue = self.imageNumber
		self.imageNumber += 1
		return '%04d' % returnValue

	def getNextPlotFile(self, suffix):
		return "images/" + self.name + "_" + self.getImageNumber() + "_" + suffix + ".png"

	def hasConstraints(self):
		return self.hasEqualityConstraints() or self.hasInequalityConstraints()


	def hasEqualityConstraints(self):
		return bool(self.h)
	def equalityConstraints(self, x):
		return self.h.evaluate(x)
	def equalityConstraintsJacobian(self, x):
		return self.h.jacobian(x)
	def getNumEqualityConstraints(self):
		if self.h is None:
			return 0
		return self.h.getOutDim()

	def hasInequalityConstraints(self):
		return bool(self.g)
	def inequalityConstraints(self, x):
		return self.g.evaluate(x)
	def inequalityConstraintsJacobian(self, x):
		return self.g.jacobian(x)
	def getNumInequalityConstraints(self):
		if self.g is None:
			return 0
		return self.g.getOutDim()

	def objective(self, x):
		return self.f.evaluate(x)
	def gradient(self, x):
		return self.f.gradient(x)
	def hessian(self, x):
		return self.f.hessian(x)

	def createBasePlotAt(self, centerX, r, title='Current Step'):
		fig = plt.figure()
		fig.set_size_inches(sys_utils.get_plot_size(), sys_utils.get_plot_size())
		ax1 = fig.add_subplot(111)

		matplotlib.rcParams['xtick.direction'] = 'out'
		matplotlib.rcParams['ytick.direction'] = 'out'

		x = linspace(centerX[0]-r, centerX[0]+r, num=100)
		y = linspace(centerX[1]-r, centerX[1]+r, num=100)
		X, Y = meshgrid(x, y)

		Z = empty((len(y), len(x)))

		plt.title(title)

		for i in range(0, len(x)):
			for j in range(0, len(y)):
				Z[j, i] = self.objective(arr([x[i], y[j]]))
		CS = plt.contour(X, Y, Z, 6, colors='k')
		plt.clabel(CS, fontsize=9, inline=1)

		for idx in range(0, self.getNumEqualityConstraints()):
			for i in range(0, len(x)):
				for j in range(0, len(y)):
					Z[j, i] = self.equalityConstraints(arr([x[i], y[j]]))[idx]
			CS = plt.contour(X, Y, Z, 6, colors='r')
			plt.clabel(CS, fontsize=9, inline=1)

		for idx in range(0, self.getNumInequalityConstraints()):
			for i in range(0, len(x)):
				for j in range(0, len(y)):
					Z[j, i] = self.inequalityConstraints(arr([x[i], y[j]]))[idx]
			CS = plt.contour(X, Y, Z, 6, colors='b')
			plt.clabel(CS, fontsize=9, inline=1)
		return ax1
