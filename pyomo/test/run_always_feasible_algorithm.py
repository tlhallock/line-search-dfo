
import numpy

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.test_objectives import Objective


numpy.set_printoptions(linewidth=255)


a = 1.0

params = AlgorithmParams()
params.x0 = numpy.array([8.0, 5.0])
params.constraints_A = numpy.array([
	[-a, 1.0],
	[+a, 1.0]
])
params.constraints_b = numpy.array([0.0, 0.0])
params.objective_function = Objective(minor_speed=1e-1, amplitude=a / 2.0, freq=2)

params.plot_bounds.append(numpy.array([-2, 0]))
params.plot_bounds.append(numpy.array([10, +5]))
params.plot_bounds.append(numpy.array([10, -5]))

always_feasible_algorithm(params)
