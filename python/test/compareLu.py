from dfo import polynomial_basis
from dfo import lagrange
from numpy import dot
from numpy import asarray
from numpy import array as arr
from numpy import zeros
from numpy import array
from numpy import multiply
from numpy import sqrt
from numpy import arctan
from numpy import reshape
from numpy import sin
from scipy.optimize import minimize
from numpy.linalg import norm
import matplotlib.pyplot as plt

from numpy import linspace
from numpy import meshgrid
from utilities import sys_utils
import matplotlib
from numpy.random import seed

from dfo.dfo_model import MultiFunctionModel

from utilities.ellipse import plotEllipse_inner

seed(1776)

tol=1e-4
n = 2
degree = 2
a = 1
xsi=1e-3
center = array((5, 0.25))
radius = 3
scale = 1.2




theConstraints = [{
	'type': 'ineq',
	'fun': lambda x: a * x[0] + x[1],
	'jac': lambda x: array((a, 1)),
}, {
	'type': 'ineq',
	'fun': lambda x: a * x[0] - x[1],
	'jac': lambda x: array((a, -1)),
}]

class objective:
	def __init__(self, minorSpeed=1e-1, amplitude=a/2, freq=2):
		self.minorSpeed = minorSpeed
		self.amplitude = amplitude
		self.freq = freq

	def evaluate(self, x):
		return 0.9 * (x[0]) + 0.1 * (self.minorSpeed * x[0] + (x[1] - self.amplitude * x[0] * sin(self.freq * x[0])) ** 2)

obj = objective()

basis = polynomial_basis.PolynomialBasis(n, degree)

class ConstraintOptions:
	def __init__(self):
		self.useEllipse = True
		self.A =  asarray([
			[a, 1],
			[a, -1]
		])
		self.b = asarray([0, 0])
		self.tol = tol

		self.ellipse_search = True

		self.line_search = False
		self.num_points_on_path = 2

		self.scale_to_original_point = False

		self.bump_xsi = False

		self.search_everything = True
		self.constraints = theConstraints

model = MultiFunctionModel([obj], basis, center, radius=radius, minXsi=1e-10, consOpts=ConstraintOptions())

def createPlot(filename, title, model, newMin=None, rho=None):
	plt.title(title)
	fig = plt.figure()
	fig.set_size_inches(sys_utils.get_plot_size(), sys_utils.get_plot_size())
	ax1 = fig.add_subplot(111)
	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'

	plt.legend(loc='lower left')

	bounds = {
		'lbX': -10, #model.currentSet[0, 0] - 2 * model.modelRadius,
		'ubX': 10, #model.currentSet[0, 0] + 2 * model.modelRadius,
		'lbY': -10, #model.currentSet[0, 1] - 2 * model.modelRadius,
		'ubY': 10, #model.currentSet[0, 1] + 2 * model.modelRadius,
		'plotArrows': True
	}
	x = linspace(bounds['lbX'], bounds['ubX'], num=100)
	y = linspace(bounds['lbY'], bounds['ubY'], num=100)
	X, Y = meshgrid(x, y)

	Z = zeros((len(y), len(x)))

	for i in range(0, len(x)):
		for j in range(0, len(y)):
			Z[j, i] = obj.evaluate(array(([x[i], y[j]])))
	CS = plt.contour(X, Y, Z, 6, colors='k')
	plt.clabel(CS, fontsize=9, inline=1)

	quad = model.getQuadraticModel(0)
	for i in range(0, len(x)):
		for j in range(0, len(y)):
			Z[j, i] = quad.evaluate(array(([x[i], y[j]])))
	CS = plt.contour(X, Y, Z, 6, colors='y')
	plt.clabel(CS, fontsize=9, inline=1)

	for con in theConstraints:
		for i in range(0, len(x)):
			for j in range(0, len(y)):
				Z[j, i] = con['fun'](arr([x[i], y[j]]))
		CS = plt.contour(X, Y, Z, 6, colors='b')
		plt.clabel(CS, fontsize=9, inline=1)

	# ax1.add_artist(plt.Circle(model.modelCenter(), model.modelRadius, color='g', fill=False))
	try:
		plotEllipse_inner(model.consOpts.ellipse, ax1, bounds, scaled=scale)
	except:
		pass
	########################################################################################################################################################
	ax1.scatter(model.currentSet[:, 0], model.currentSet[:, 1], s=20, c='r', marker="x", label='poised set')
	if newMin is not None:
		ax1.scatter(array((newMin[0])), array((newMin[1])), s=20, c='g', marker="x", label='poised set')
		if rho is not None:
			ax1.text(newMin[0], newMin[1], 'rho' + str(rho))
	#
	# ax1.axis([center[0] - 2 * radius, center[0] + 2 * radius, center[1] - 2 * radius, center[1] + 2 * radius])

	lmbdaStr = "Max Lambda = "
	if model.cert.Lambda is not None:
		lmbdaStr += str(max(model.cert.Lambda))
	else:
		lmbdaStr += "undefined"

	lmbdaStr = lmbdaStr + ", Max Constrained Lambda = "
	if model.cert.LambdaConstrained is not None:
		lmbdaStr += str(max(model.cert.LambdaConstrained))
	else:
		lmbdaStr += "undefined"

	ax1.text(model.modelCenter()[0], model.modelCenter()[1], lmbdaStr)

	fig.savefig(filename)
	plt.close()

#model.improve()

iteration = 0
while True:
	iteration += 1
	print('function evaluation: ' + str(model.functionEvaluations))

	if norm(model.modelCenter()) < tol and model.modelRadius < tol:
		print(model.modelRadius)
		break

	if not model.improve():
		if model.phi is not None:
			createPlot(
				filename='images/iteration_' + str(iteration) + '_unableToModel.png',
				title='Iteration ' + str(iteration),
				model=model
			)
		model.multiplyRadius(0.5)
		continue

	quad = model.getQuadraticModel(0)

	rscale = scale
	if model.consOpts.scale_to_original_point:
		rscale = model.consOpts.ellipse['include_point_scale']
		if rscale > 1:
			print("this happens")
	constraintsCopy = theConstraints + [{
		'type': 'ineq',
		'fun': model.consOpts.ellipse['scaled_fun'](rscale),
		'jac': model.consOpts.ellipse['scaled_jac'](rscale),
	}]
	minimumResult = minimize(quad.evaluate, jac=quad.gradient, x0=0.5 * model.modelCenter() + 0.5 * center,
						constraints=constraintsCopy, method='SLSQP', options={"disp": False, "maxiter": 1000}, tol=tol / 4)
	if not minimumResult.success:
		print('unable to solve trust region problem')
	trialPoint = minimumResult.x
	newVal, called = model.computeValueFromDelegate(trialPoint)
	oldVal = obj.evaluate(model.modelCenter())
	oldValM = quad.evaluate(model.modelCenter())

	rho = (oldVal - newVal) / (oldValM - quad.evaluate(trialPoint))

	createPlot(filename='images/iteration_' + str(iteration) + '_new_point.png', title='Iteration ' + str(iteration), model=model, newMin=trialPoint, rho=rho)

	print('rho', rho)
	if rho < .5:
		print('decreasing radius')
		model.multiplyRadius(0.5)
		continue


	print("the old model center", model.trust_region_center)
	print("the new model center", trialPoint)
	print("the difference", norm(model.trust_region_center - trialPoint))
	#print("compared to ", model.modelRadius / 8)
	delta = norm(model.trust_region_center - trialPoint)
	if delta < model.modelRadius / 8:
		if delta < tol:
			continue
		model.multiplyRadius(.5)
		continue

	print('new function value:' + str(newVal))
	if minimumResult.success:
		model.setNewModelCenter(trialPoint)

print("iterations")
print(iteration)

print("Function evaluations")
print(model.functionEvaluations)