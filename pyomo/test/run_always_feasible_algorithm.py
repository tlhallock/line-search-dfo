import traceback
import numpy
import os

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.util.test_objectives import Objective

numpy.set_printoptions(linewidth=255)


params = [{
#	'trust_region_options': {
#		'shape': 'circle',
#		'search': 'none',
#	},
#	'directory': 'circle',
#	'increase-radius': True,
#	'replacement-strategy-params': {
#		'strategy': 'fixed-xsi',
#	},
#	'basis': 'quadratic',
#}, {
#	'trust_region_options': {
#		'shape': 'ellipse',
#		'include_as_constraint': True,
#		'search': 'none',
#	},
#	'directory': 'ellipse',
#	'increase-radius': True,
#	'replacement-strategy-params': {
#		'strategy': 'fixed-xsi',
#	},
#	'basis': 'quadratic',
#}, {
#	'trust_region_options': {
#		'shape': 'circle',
#		'search': 'anywhere',
#	},
#	'directory': 'circle_anywhere',
#	'increase-radius': True,
#	'replacement-strategy-params': {
#		'strategy': 'fixed-xsi',
#	},
#	'basis': 'quadratic',
#}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'anywhere',
	},
	'directory': 'ellipse_anywhere',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
#}, {
#	'trust_region_options': {
# 		'shape': 'scaled-ellipse',
# 		'search': 'anywhere',
# 	},
# 	'directory': 'scaled_ellipse_anywhere',
# 	'increase-radius': False,
# 	'replacement-strategy-params': {
# 		'strategy': 'fixed-xsi',
# 	},
# 	'basis': 'quadratic',
# }, {
# 	'trust_region_options': {
# 		'shape': 'scaled-ellipse',
# 		'search': 'segment',
# 		'number_of_points': 1,
# 	},
# 	'directory': 'scaled_ellipse_segment_1',
# 	'increase-radius': False,
# 	'replacement-strategy-params': {
# 		'strategy': 'fixed-xsi',
# 	},
# 	'basis': 'quadratic',
# }, {
# 	'trust_region_options': {
# 		'shape': 'scaled-ellipse',
# 		'search': 'segment',
# 		'number_of_points': 2,
# 	},
# 	'directory': 'scaled_ellipse_segment_2',
# 	'increase-radius': False,
# 	'replacement-strategy-params': {
# 		'strategy': 'fixed-xsi',
# 	},
# 	'basis': 'quadratic',
#}, {
#	'trust_region_options': {
#		'shape': 'circle',
#		'search': 'segment',
#		'number_of_points': 1,
#	},
#	'directory': 'circle_segment_1',
#	'increase-radius': True,
#	'replacement-strategy-params': {
#		'strategy': 'fixed-xsi',
#	},
#	'basis': 'quadratic',
#}, {
#	'trust_region_options': {
#		'shape': 'ellipse',
#		'include_as_constraint': True,
#		'search': 'segment',
#		'number_of_points': 1,
#	},
#	'directory': 'ellipse_segment_1',
#	'increase-radius': True,
#	'replacement-strategy-params': {
#		'strategy': 'fixed-xsi',
#	},
#	'basis': 'quadratic',
#}, {
#	'trust_region_options': {
# 		'shape': 'circle',
# 		'search': 'segment',
# 		'number_of_points': 2,
# 	},
# 	'directory': 'circle_segment_2',
#	'increase-radius': True,
#	'replacement-strategy-params': {
#		'strategy': 'fixed-xsi',
#	},
#	'basis': 'quadratic',
# }, {
#	'trust_region_options': {
#		'shape': 'ellipse',
#		'include_as_constraint': True,
#		'search': 'segment',
#		'number_of_points': 2,
#	},
#	'directory': 'ellipse_segment_2',
#	'increase-radius': False,
#	'replacement-strategy-params': {
#		'strategy': 'fixed-xsi',
#	},
#	'basis': 'quadratic',
# }, {
#	'trust_region_options': {
#		'shape': 'polyhedral',
#		'search': 'none',
#	},
#	'directory': 'feasible_intersect_trust',
#	'increase-radius': False,
#	'replacement-strategy-params': {
#		'strategy': 'adaptive-xsi',
#	},
#	'basis': 'quadratic',
}]


# to add shift xsi
# to add scale
# to add 

for p in params:
	try:
		a = 0.5

		params = AlgorithmParams()
		params.basis_type = p['basis']
		params.x0 = numpy.array([3.0, 0.5])
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

