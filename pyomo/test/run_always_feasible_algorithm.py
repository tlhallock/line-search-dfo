
import numpy
import os

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.test_objectives import Objective

numpy.set_printoptions(linewidth=255)


params = [{
	'trust_region_options': {
		'shape': 'circle',
		'search': 'none',
	},
	'directory': 'circle',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'none',
	},
	'directory': 'ellipse',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'anywhere',
	},
	'directory': 'circle_anywhere',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'anywhere',
	},
	'directory': 'ellipse_anywhere',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'circle_segment_1',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'ellipse_segment_1',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'circle_segment_2',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'ellipse_segment_2',
}]


# to add shift xsi
# to add scale
# to add 

for p in params:
	try:
		a = 0.5

		params = AlgorithmParams()
		params.x0 = numpy.array([3.0, 0.5])
		params.constraints_A = numpy.array([
			[-a, +1.0],
			[-a, -1.0]
		])
		params.directory = p['directory']
		params.trust_region_strategy_params = p['trust_region_options']
		params.constraints_b = numpy.array([0.0, 0.0])
		params.objective_function = Objective(minor_speed=1e-1, amplitude=a / 2.0, freq=2)

		params.plot_bounds.append(numpy.array([-2, 0]))
		params.plot_bounds.append(numpy.array([10, +5]))
		params.plot_bounds.append(numpy.array([10, -5]))

		output_path = 'images/{}'.format(params.directory)
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		always_feasible_algorithm(params)
	except:
		print("failed")

