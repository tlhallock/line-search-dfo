import traceback
import numpy
import os
import numpy as np

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.test_objectives import Objective

import sys
sys.path.append('/work/research/highly_constrained/mystic/build/lib')
from mystic.models.schittkowski import Paviani


class PavianiObjective:
	def __init__(self):
		self.paviani = Paviani(2)

	def evaluate(self, x):
		return self.paviani.function(x)


numpy.set_printoptions(linewidth=255)

params = AlgorithmParams()
params.basis_type ='quadratic'
params.x0 = numpy.array([4.0, 0.2])
params.constraints_A = numpy.array([
	[-1, +1.0],
	[-1, -1.0]
])
params.constraints_b = numpy.array([0.0, 0.0])


params.directory = 'general'
params.trust_region_strategy_params = {
	'shape': 'ellipse',
	'search': 'none',
}
params.objective_function = PavianiObjective()
params.radius_increase_factor = 1.0

params.plot_bounds.append(numpy.array([-2, 0]))
params.plot_bounds.append(numpy.array([10, +5]))
params.plot_bounds.append(numpy.array([10, -5]))

params.point_replacement_params = {
	'strategy': 'fixed-xsi',
}

output_path = 'images/general/'
if not os.path.exists(output_path):
	os.makedirs(output_path)

always_feasible_algorithm(params)
