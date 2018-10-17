from trust_region.algorithm.tr_search.shape.circle import get_circular_trust_region_objective
from trust_region.algorithm.tr_search.shape.ellipse import get_elliptical_trust_region_objective
from trust_region.algorithm.tr_search.shape.polyhedron import get_polyhedral_trust_region_objective
from trust_region.algorithm.tr_search.shape.scaled_ellipse import get_scaled_elliptical_trust_region_objective

from trust_region.algorithm.tr_search.searches.no_search import no_search
from trust_region.algorithm.tr_search.searches.polyhedron_pattern_search import search_anywhere
from trust_region.algorithm.tr_search.searches.segment_search import search_segment


def parse_tr_strategy(params):
	options = {}

	if params['shape'] == 'circle':
		shape = get_circular_trust_region_objective
	elif params['shape'] == 'ellipse':
		shape = get_elliptical_trust_region_objective
		options['must_include_center'] = params['search'] != 'none'
	elif params['shape'] == 'polyhedral':
		shape = get_polyhedral_trust_region_objective
	elif params['shape'] == 'scaled-ellipse':
		shape = get_scaled_elliptical_trust_region_objective
	else:
		raise Exception('unknown trust region shape: {}'.format(params['shape']))

	if params['search'] == 'segment':
		search = search_segment
		options['number_of_points'] = params['number_of_points']
		options['num_trial_points'] = 5
	elif params['search'] == 'anywhere':
		options['random-starting-points'] = 5 if params['shape'] == 'circle' else 1
		options['random-search-directions'] = 10 if params['shape'] == 'circle' else 2
		search = search_anywhere
	elif params['search'] == 'none':
		search = no_search
	else:
		raise Exception('unknown search type {}'.format(params['search']))

	def find_trust_region(context):
		objective_value, plot_details = search(
			context,
			shape,
			options
		)
		return objective_value.success, objective_value.trust_region, plot_details
	return find_trust_region

