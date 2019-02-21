import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy


center = numpy.array([
	0.0015547982842035778,
	1.2023159251392687e-06
])
A = numpy.array([
	[-0.5, 1.0],
	[-0.5, -1.0],
	[1.0, 0.0],
	[-1.0, 0.0],
	[0.0, 1.0],
	[0.0, -1.0]
])
b = numpy.array([
	0.0,
	0.0,
	0.00719250839786401,
	0.004082911829456855,
	0.005638912429585571,
	0.005636507797735293
])
include_point = numpy.array([
	0.0015547982842035778,
	1.2023159251392687e-06
])
tolerance = 1e-08


bbar = b - numpy.dot(A, center)
# scale
k = 1.0 / min(abs(bbar))
bbar = k * bbar

model = ConcreteModel()
model.q = Var(range(3))

model.constraints = ConstraintList()

for i in range(A.shape[0]):
	model.constraints.add(
		A[i, 0] ** 2 * model.q[0] +
		2 * A[i, 0] * A[i, 1] * model.q[1] +
		A[i, 1] ** 2 * model.q[2] <= bbar[i] * bbar[i] / 2
	)

si = include_point - center
k2 = k * k
model.constraints.add(
	-si[0] * model.q[1] * si[1] + si[0] * model.q[2] * si[0] +
	si[1] * model.q[0] * si[1] - si[1] * model.q[1] * si[0] -
	2 * model.q[0] * model.q[2] * k2 + 2 * model.q[1] * model.q[1] * k2 <= 0
)

model.constraints.add(model.q[0] >= 0.0)
model.constraints.add(model.q[2] >= 0.0)

model.objective = Objective(expr=model.q[0] * model.q[2] - model.q[1] * model.q[1], sense=maximize)

#opt = SolverFactory('ipopt', executable='/home/thallock/anaconda3/envs/trust_region/bin/ipopt')
opt = SolverFactory('cplex', executable='/home/thallock/Applications/cplex/cplex/bin/x86-64_linux/cplex')
result = opt.solve(model)
print(result)
ok = result.solver.status == SolverStatus.ok
if not ok:
	print("warning solver did not return ok")
optimal = result.solver.termination_condition == TerminationCondition.optimal
if not optimal:
	print("warning solver did not return optimal")
