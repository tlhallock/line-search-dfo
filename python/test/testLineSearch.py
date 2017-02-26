



from dfo import polynomial_basis
from dfo import dfo_model
from numpy import array as arr
from numpy import zeros
from numpy.linalg import norm
from utilities import functions
from numpy import random
import unittest

import program

from scipy.optimize import minimize




Q = arr([[1, 0],[0, 5]])
b = arr([0, 0])
#b = arr([1, 1])
c=0


# write tests

q = functions.Quadratic(Q, b, c)
c1 = functions.Line(arr([1,  1]), -1)
#c2 = functions.Line(arr([1, -1]), 10)
c2 = functions.Wiggles()

equality = []
equality.append(c1)

inequality = []
inequality.append(c2)

x0 = arr([1, 2])

#statement = program.Program(q, equality, inequality, x0)
statement = program.DfoProgram("wiggles", q, equality, inequality, x0, plotImprovements=True)

cons=({'type':'eq','fun': statement.equalityConstraints, 'jac': statement.equalityConstraintsJacobian},
	  {'type': 'ineq', 'fun': statement.inequalityConstraints, 'jac': statement.inequalityConstraintsJacobian})

#hess=statement.hessian,
res = minimize(statement.objective, x0, method='SLSQP', jac=statement.gradient, constraints=cons, options={'xtol': 1e-8, 'disp': False, 'maxfev': 1000})

options={'xtol': 1e-8, 'disp': False, 'maxfev': 1000}

print(res)


#
#
# class TestModel(unittest.TestCase):
# 	def __init__(self, *args, **kwargs):
# 		super(TestModel, self).__init__(*args, **kwargs)
# all([not newActive[i] for i, x in enumerate(state.active) if not x])
#
# 	def test_simpleValuesMatch(self):
# 		x = arr([3, 4])
#
#
#
