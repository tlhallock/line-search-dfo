

from numpy import array as arr
from utilities import functions
from algorithms import trust_filter
from utilities import sys_utils
import program


sys_utils.clean_images_directory()



Q = arr([[1, 0],[0, 5]])
b = arr([0, 0])
c=0

q = functions.Quadratic(Q, b, c)
c1 = functions.Line(arr([1,  1]), -1)

equality = functions.VectorFunction(2)
equality.add(functions.Wiggles())

inequality = functions.VectorFunction(2)
inequality.add(functions.Line(arr([1, -1]), 10))

x0 = arr([1, 2])


program = program.Program("wiggles", q, equality, inequality, x0)

constants = trust_filter.Constants()
results = trust_filter.trust_filter(program, constants)

print(results)
print(results.x_min)
print(results.f_min)




#cons=({'type':'eq','fun': statement.equalityConstraints, 'jac': statement.equalityConstraintsJacobian},
#	  {'type': 'ineq', 'fun': statement.inequalityConstraints, 'jac': statement.inequalityConstraintsJacobian})
#
#hess=statement.hessian,
#res = minimize(statement.objective, x0, method='SLSQP', jac=statement.gradient, constraints=cons, options={'xtol': 1e-8, 'disp': False, 'maxfev': 1000})
#
#options={'xtol': 1e-8, 'disp': False, 'maxfev': 1000}
#
#print(res)


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
