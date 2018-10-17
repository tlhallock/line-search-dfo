import abc

import numpy
import copy

from trust_region.optimization.maximize_lagrange import maximize_lagrange_quadratic


def _find_best_replacement(check):
	return maximize_lagrange_quadratic(check.basis, check.trust_region, check.basis_coefficients)


class ReplacementCheck:
	def __init__(self, basis, trust_region, max_value, current_point, basis_coefficients):
		self.basis = basis
		self.trust_region = trust_region
		self.max_value = max_value
		self.current_point = current_point
		self.basis_coefficients = basis_coefficients

		self.should_replace = False
		self.new_point = None


def replace_far_points(check, options):
	far_radius = options['far-radius']
	if numpy.linalg.norm(check.current_point) > far_radius:
		check.should_replace = True


def do_not_replace(check, options):
	check.should_replace = False


def fixed_xsi(check, options):
	xsi = options['xsi']
	if abs(check.max_value) > xsi:
		return

	check.should_replace = True
	alternate_solution = _find_best_replacement(check)
	if alternate_solution.objective < xsi:
		raise Exception('Even after replacement, the new point is not poised')
	check.new_point = alternate_solution.x


def always_replace(check, options):
	check.should_replace = True
	check.new_point = _find_best_replacement(check).x


def finisher(check, options):
	if not check.should_replace:
		return
	if check.new_point is not None:
		return
	check.new_point = _find_best_replacement(check).x


def adaptive_xsi(check, options):
	current_xsi = options['current-xsi']
	minimum_xsi = options['minimum-xsi']
	keep_tolerance = options['keep-tolerance']

	if check.should_replace:
		return

	if check.max_value > current_xsi:
		return

	alternate_solution = _find_best_replacement(check)
	check.new_point = alternate_solution.x
	if check.max_value / keep_tolerance > alternate_solution.objective:
		# Not enough possible improvement
		return

	check.should_replace = True
	if alternate_solution.objective < minimum_xsi:
		raise Exception('Even after replacement, the new point is not poised')


REPLACEMENT_CHECKS = {
	'fixed-xsi': {
		'checkers': [replace_far_points, fixed_xsi, finisher],
		'options': {'far-radius': 1.5, 'xsi': 1e-4}
	},
	'adaptive-xsi': {
		'checkers': [replace_far_points, adaptive_xsi, finisher],
		'options': {'far-radius': 1.5, 'current-xsi': 1e-4, 'minimum-xsi': 1e-10, 'keep-tolerance': 1.0}
	},
	'always-replace': {
		'checkers': [always_replace, finisher],
		'options': {}
	},
	'never-replace': {
		'checkers': [do_not_replace],
		'options': {}
	}
}


def parse_replacement_policy(parameters):
	obj = REPLACEMENT_CHECKS[parameters['strategy']]
	return obj['checkers'], copy.deepcopy(obj['options'])








