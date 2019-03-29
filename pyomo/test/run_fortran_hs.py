import traceback
import numpy
import os
import json
import numpy as np
import time
import sys

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.hs_fortran_caller import HsProblem
from trust_region.util.utils import write_json
from trust_region.util.plots import create_plot_on

numpy.set_printoptions(linewidth=255)


def run_on_objective(objective, tr_strategy):
	try:
		print("=======================================")
		print('Problem', objective, tr_strategy['name'])

		problem = HsProblem(objective)
		write_json(problem.to_json(), sys.stdout)

		if tr_strategy['params']['search'] == 'segment' and problem.n < tr_strategy['params']['number_of_points']:
			return

		params = AlgorithmParams()
		params.basis_type = 'quadratic'
		params.buffer_factor = tr_strategy['buffer-factor']

		params.constraints_polyhedron = problem.constraints
		for idx, lbi in enumerate(problem.lb):
			if np.isinf(lbi):
				continue
			params.constraints_polyhedron = params.constraints_polyhedron.add_lb(idx, lbi)
		for idx, ubi in enumerate(problem.ub):
			if np.isinf(ubi):
				continue
			params.constraints_polyhedron = params.constraints_polyhedron.add_ub(idx, ubi)

		feasibility = problem.get_initial_feasibility()
		if feasibility == 'infeasible':
			print('Problem does not have a feasible start')
			params.x0 = params.constraints_polyhedron.get_feasible_point()
		elif feasibility == 'active' and tr_strategy['requires-interior-x0']:
			print('Problem has active constraint, changing starting point')
			params.x0 = params.constraints_polyhedron.get_feasible_point()
			return
		else:
			params.x0 = problem.x0

		params.directory = 'hs_' + str(objective) + '_' + tr_strategy['name']
		params.trust_region_strategy_params = tr_strategy['params']

		class Objective:
			def evaluate(self, x):
				return problem.evaluate_objective(x)

		params.objective_function = Objective()
		params.radius_increase_factor = 1.2

		params.point_replacement_params = {
			'strategy': 'far-fixed-xsi',
		}

		if not os.path.exists('images/' + params.directory):
			os.mkdir('images/' + params.directory)

		result = always_feasible_algorithm(params)
		if not result['success']:
			print(result['stack trace'])
		with open(os.path.join('images', params.directory, 'result.json'), 'w') as results_out:
			write_json(
				{
					'result': result,
					'problem': problem.to_json()
				},
				results_out
			)
	except:
		traceback.print_exc()


OBJECTIVES = [
	21,
	224,
	231,
	232,
	24,
	25,
	35,
	36,
	37,
	44,
	45,
	76,
	250,
	251
]
for objective in OBJECTIVES:
	for tr_strategy in [{
		# 'name': 'ellipse',
		# 'params': {
		# 	'shape': 'ellipse',
		# 	'search': 'none',
		# },
		# 'requires-interior-x0': True,
		# 'buffer-factor': 0.75
	# }, {
	# 	'name': 'ellipse_everywhere',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'anywhere',
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
	# 	'name': 'ellipse_segment_1',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 1,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
		'name': 'ellipse_segment_2',
		'params': {
			'shape': 'ellipse',
			'search': 'segment',
			'number_of_points': 2,
		},
		'requires-interior-x0': False,
		'buffer-factor': None
	}, {
		'name': 'ellipse_segment_3',
		'params': {
			'shape': 'ellipse',
			'search': 'segment',
			'number_of_points': 3,
		},
		'requires-interior-x0': False,
		'buffer-factor': None
	}, {
		'name': 'ellipse_segment_4',
		'params': {
			'shape': 'ellipse',
			'search': 'segment',
			'number_of_points': 4,
		},
		'requires-interior-x0': False,
		'buffer-factor': None
	}, {
		'name': 'ellipse_segment_5',
		'params': {
			'shape': 'ellipse',
			'search': 'segment',
			'number_of_points': 5,
		},
		'requires-interior-x0': False,
		'buffer-factor': None
	# }, {
		# 'name': 'polyhedral',
		# 'params': {
		# 	'shape': 'polyhedral',
		# 	'search': 'anywhere',
		# },
		# 'requires-interior-x0': False,
		# 'buffer-factor': None
	# }, {
	# 	'name': 'circumscribed_ellipse',
	# 	'params': {
	# 		'shape': 'circumscribed-ellipse',
	# 		'search': 'none',
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	}]:
		run_on_objective(objective, tr_strategy)



