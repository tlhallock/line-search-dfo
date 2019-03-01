import traceback
import numpy
import os
import json
import numpy as np

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.polyhedron import Polyhedron
from trust_region.util.utils import write_json
from trust_region.util.hs import OBJECTIVES

numpy.set_printoptions(linewidth=255)

#
# from trust_region.util.polyhedron import get_polyhedron
# import sys
# polys = [
# 	(lambda pnts: { 'p': get_polyhedron(pnts).to_json(), 'pnts': pnts }) (5 * numpy.random.random((10, 2)))
# 	for i in range(3)
# ] + [
# 	(lambda pnts: { 'p': get_polyhedron(pnts).to_json(), 'pnts': pnts }) (5 * numpy.random.random((10, 3)))
# 	for i in range(2)
# ] + [
# 	(lambda pnts: { 'p': get_polyhedron(pnts).to_json(), 'pnts': pnts }) (5 * numpy.random.random((20, 7)))
# 	for i in range(2)
# ]
# write_json(polys, sys.stdout)


def run_on_objective(objective, tr_strategy):
	try:
		print('Problem', objective['problem'], tr_strategy['name'])
		if not objective['feasible']:
			print('Problem does not have a feasible start')
			return

		params = AlgorithmParams()
		params.basis_type = 'quadratic'
		params.buffer_factor = 0.1

		params.constraints_polyhedron = Polyhedron(
			objective['constraints'][:, :-1],
			objective['constraints'][:, -1:].flatten()
		)
		for idx, row in enumerate(objective['bounds']):
			if not np.isinf(row[0]):
				params.constraints_polyhedron = params.constraints_polyhedron.add_lb(idx, row[0])
			if not np.isinf(row[1]):
				params.constraints_polyhedron = params.constraints_polyhedron.add_ub(idx, row[1])

		if tr_strategy['requires-interior-x0'] and min(abs((params.constraints_polyhedron.evaluate(objective['x0'])))) < 1e-12:
			print('Problem starts with an active constraint.')
			return

		params.x0 = objective['x0']

		params.directory = 'hs_' + str(objective['problem']) + '_' + tr_strategy['name']
		params.trust_region_strategy_params = tr_strategy['params']

		class Objective:
			def evaluate(self, x):
				return objective['objective'](x)

		params.objective_function = Objective()
		params.radius_increase_factor = 1.2

		params.point_replacement_params = {
			'strategy': 'far-fixed-xsi',
		}

		if not os.path.exists('images/' + params.directory):
			os.mkdir('images/' + params.directory)

		result = always_feasible_algorithm(params)

		expected_minimum = objective['minimum']
		expected_minimizer = objective['minimizer']
	except:
		traceback.print_exc()


for objective in OBJECTIVES:
	for tr_strategy in [{
		'name': 'ellipse',
		'params': {
			'shape': 'ellipse',
			'search': 'none',
		},
		'requires-interior-x0': False
	}, {
		'name': 'ellipse_everywhere',
		'params': {
			'shape': 'ellipse',
			'search': 'anywhere',
		},
		'requires-interior-x0': False
	}, {
		'name': 'polyhedral',
		'requires-interior-x0': False,
		'params': {
			'shape': 'polyhedral',
			'search': 'anywhere',
		}
	}]:
		run_on_objective(objective, tr_strategy)
