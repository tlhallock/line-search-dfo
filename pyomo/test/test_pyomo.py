
from pyomo.core import *
from pyomo.opt import *
import pyomo.environ

model = ConcreteModel()

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)


def constraint_rule(model):
    return model.x + model.y >= 10


model.constraint = Constraint(rule=constraint_rule)


def objective_rule(model):
    coefficients = [0, 0, 0, 0, 0, 0]
    return (
        1.0 * coefficients[0] +
        1.0 * coefficients[1] * model.x[0] +
        1.0 * coefficients[2] * model.x[1] +
        0.5 * coefficients[3] * model.x[0] * model.x[0] +
        0.5 * coefficients[4] * model.x[1] * model.x[0] +
        0.5 * coefficients[5] * model.x[1] * model.x[1]
    )


model.objective = Objective(rule=objective_rule, sense=minimize)

opt = SolverFactory('ipopt')
opt.solve(model)

for v in model.component_data_objects(Var):
    print(str(v), v.value)
