import traceback
import numpy
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.test_objectives import Objective

from test.run_params import params

numpy.set_printoptions(linewidth=255)

# to add shift xsi
# to add scale
# to add


def run_always_feasible(p):
	try:
		a = 0.1

		params = AlgorithmParams()
		params.basis_type = p['basis']
		params.x0 = numpy.array([4.0, 0.2])
		params.constraints_A = numpy.array([
			[-a, +1.0],
			[-a, -1.0]
		])
		params.directory = p['directory']
		params.trust_region_strategy_params = p['trust_region_options']
		params.constraints_b = numpy.array([0.0, 0.0])
		params.objective_function = Objective(minor_speed=1e-1, amplitude=a / 2.0, freq=2)
		if not p['increase-radius']:
			params.radius_increase_factor = 1.0

		params.plot_bounds.append(numpy.array([-2, 0]))
		params.plot_bounds.append(numpy.array([10, +5]))
		params.plot_bounds.append(numpy.array([10, -5]))

		params.point_replacement_params = p['replacement-strategy-params']

		output_path = 'images/{}'.format(params.directory)
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		always_feasible_algorithm(params)
	except Exception as e:
		print("failed")
		print(e)
		traceback.print_exc()


num_threads = 0

if num_threads > 0:
	pool = ThreadPool(num_threads)
	pool.map(run_always_feasible, params)
else:
	for p in params:
		run_always_feasible(p)
