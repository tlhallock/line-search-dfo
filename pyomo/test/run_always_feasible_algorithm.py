
import numpy

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.test_objectives import Objective

numpy.set_printoptions(linewidth=255)


trust_region_params = [{
		'trust_region_options': {
	'shape': 'circle',
	'search': 'none',

	},
}, {

		'trust_region_options': {
	'shape': 'ellipse',
	'include_as_constraint': True,
	'search': 'none',


	},
}, {

		'trust_region_options': {

			'shape': 'circle',
			'search': 'anywhere',

	},
}, {

		'trust_region_options': {



	'shape': 'ellipse',
	'include_as_constraint': True,
	'search': 'anywhere',
		},
}, {

		'trust_region_options': {
	'shape': 'circle',
	'search': 'segment',
	'number_of_points': 1,


	}
}, {

		'trust_region_options': {

			'shape': 'ellipse',
			'include_as_constraint': True,
			'search': 'segment',
			'number_of_points': 1,
		},

}, {

		'trust_region_options': {

			'shape': 'circle',
			'search': 'segment',
			'number_of_points': 2,
		},

}, {

		'trust_region_options': {

			'shape': 'ellipse',
			'include_as_constraint': True,
			'search': 'segment',
			'number_of_points': 2,
		},

}]

# to add shift xsi
# to add scale
# to add 


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
