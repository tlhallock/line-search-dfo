from numpy import array
from numpy import dot
from numpy import reshape
from numpy import zeros
from numpy.linalg import norm
from scipy.optimize import minimize


cIneq = array([ 9.])
AIneq = array([[ 1., -1.]])

cEq = array([ 0.09070257])
AEq = array([[ 1.00427046,  0.35586232]])

radius = 50

cons = [
	{'type': 'ineq',
		'fun': lambda n: radius**2 - dot(n, n),
		'jac': lambda n: reshape(-2*n, (1,2))},
	{'type': 'ineq',
		'fun': lambda n: -cIneq - dot(AIneq, n),
		'jac': lambda n: -AIneq},
	{'type': 'eq',
		'fun': lambda n: cEq + dot(AEq, n),
		'jac': lambda n: AEq}]

res = minimize(lambda n: dot(n, n), jac=lambda n: reshape(2 * n, (1,2)), x0=zeros(2),
					constraints=cons, method='SLSQP', options={"disp": False, "maxiter": 1000}, tol=1e-8)

print((radius**2 - dot(res.x, res.x))**(.5)) # -11.5640205848
print(res.success) # True

def dbl_check_sol(cons, res):
	if not res.success:
		return True
	for c in cons:
		if c['type'] == 'ineq':
			if (c['fun'](res.x) < -1e-8).any():
				return False
		elif c['type'] == 'eq':
			if norm(c['fun'](res.x)) > 1e-8:
				return False
		else:
			raise Exception('unknown type of constraint')
	return True

print(dbl_check_sol(cons, res)) # False
print(res) # I see no other errors, and the message is 'Optimization terminated successfully.'
