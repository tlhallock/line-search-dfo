import traceback
import numpy
import os
import json
import numpy as np

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm

from trust_region.util.wikipedia_objectives import WIKIPEDIA_OBJECTIVES

numpy.set_printoptions(linewidth=255)


def translate(A, b, center):
	# Ax \le b
	# A(x - center) \le b
	# Ax \le b + A*center
	return A, b + numpy.dot(A, center)


def rotate(A, b, theta):
	rotation = numpy.array([
		[+numpy.cos(theta), -numpy.sin(theta)],
		[+numpy.sin(theta), +numpy.cos(theta)]
	])
	# ARx \le b
	return numpy.dot(A, rotation), b


def intersection(A1, b1, A2, b2):
	return numpy.append(A1, A2, axis=0), numpy.append(b1, b2)


def run_on_objective(objective):
	print(objective['name'])
	output_path = 'images/' + objective['name']
	output_file_path = os.path.join(output_path, 'minimizer.json')

	if os.path.exists(output_file_path):
		print('\talready done')
		return
	if objective['lb'] is None or objective['ub'] is None:
		print('\tunbounded')
		return

	params = AlgorithmParams()
	params.basis_type = 'quadratic'

	wedge_a, wedge_b = numpy.array([
		[-1, +1.0],
		[-1, -1.0]
	]), numpy.array([0.0, 0.0])
	wedge_a, wedge_b = rotate(wedge_a, wedge_b, numpy.random.random() * 2 * numpy.pi)
	wedge_a, wedge_b = translate(wedge_a, wedge_b, objective['minimizer'])
	params.constraints_A, params.constraints_b = intersection(
		wedge_a,
		wedge_b,
		numpy.array([
			[+1, +0],
			[-1, +0],
			[+0, +1],
			[+0, -1],
		]),
		numpy.array([
			+objective['ub'][0],
			-objective['lb'][0],
			+objective['ub'][1],
			-objective['lb'][1],
		])
	)

	print('searching for feasible start')
	while True:
		params.x0 = np.random.uniform(objective['lb'], objective['ub'])
		if (numpy.dot(params.constraints_A, params.x0) <= params.constraints_b).all():
			break
	print('starting at ', params.x0)

	params.directory = objective['name']
	params.trust_region_strategy_params = {
		'shape': 'ellipse',
		'search': 'none',
	}

	class Objective:
		def evaluate(self, x):
			return objective['func'](x)

	params.objective_function = Objective()
	params.radius_increase_factor = 1.2

	params.plot_bounds.append(objective['lb'])
	params.plot_bounds.append(objective['ub'])
	# params.plot_bounds.append(objective['minimizer'])

	params.point_replacement_params = {
		'strategy': 'fixed-xsi',
	}

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	result = always_feasible_algorithm(params)
	with open(output_file_path, 'w') as outfile:
		json.dump({
			'found': {
				'minimum': float(result['minimum']),
				'minimizer': [float(xi) for xi in result['minimizer']]
			},
			'expected': {
				'minimum': float(objective['minimum']),
				'minimizer': [float(xi) for xi in objective['minimizer']]
			}
		}, outfile, indent=2)

	print('found', result['minimizer'], '=>', result['minimum'])
	print('expected', objective['minimizer'], '=>', objective['minimum'])


def get_objective_by_name(name):
	matching = [o for o in WIKIPEDIA_OBJECTIVES if o['name'] == name]
	if len(matching) != 1:
		raise Exception('bad name: ' + name)
	return matching[0]


# print(get_objective_by_name('McCormick')['gradient'](np.array([
# 	2.59444544,
# 	1.59441382
# ])))

# run_on_objective(np.random.choice(WIKIPEDIA_OBJECTIVES))
# run_on_objective(get_objective_by_name('McCormick'))
# run_on_objective(get_objective_by_name('Goldstein-Price'))
for objective in WIKIPEDIA_OBJECTIVES:
	try:
		run_on_objective(objective)
	except:
		pass

