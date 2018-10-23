
params = [{
	'trust_region_options': {
		'shape': 'circle',
		'search': 'none',
	},
	'directory': 'circle',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'none',
	},
	'directory': 'ellipse',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'anywhere',
	},
	'directory': 'circle_anywhere',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
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
}, {
	'trust_region_options': {
		'shape': 'scaled-ellipse',
		'search': 'anywhere',
	},
	'directory': 'scaled_ellipse_anywhere',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'scaled-ellipse',
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'scaled_ellipse_segment_1',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'scaled-ellipse',
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'scaled_ellipse_segment_2',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'circle_segment_1',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'ellipse_segment_1',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'circle_segment_2',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'ellipse_segment_2',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'polyhedral',
		'search': 'none',
	},
	'directory': 'feasible_intersect_trust',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'adaptive-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'simplex',
		'search': 'simplex-search',
	},
	'directory': 'simplex',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'circumscribed-ellipse',
		'search': 'none',
	},
	'directory': 'circumscribed_ellipse',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'anywhere',
		'heuristics': {
			'num': 30,
		}
	},
	'directory': 'ellipse_anywhere_heuristic',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'quadratic',
}, {
	####################################################################################################################
	'trust_region_options': {
		'shape': 'circle',
		'search': 'none',
	},
	'directory': 'circle_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'none',
	},
	'directory': 'ellipse_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'anywhere',
	},
	'directory': 'circle_anywhere_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'anywhere',
	},
	'directory': 'ellipse_anywhere_linear',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'scaled-ellipse',
		'search': 'anywhere',
	},
	'directory': 'scaled_ellipse_anywhere_linear',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'scaled-ellipse',
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'scaled_ellipse_segment_1_linear',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'scaled-ellipse',
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'scaled_ellipse_segment_2_linear',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'circle_segment_1_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'segment',
		'number_of_points': 1,
	},
	'directory': 'ellipse_segment_1_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'circle',
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'circle_segment_2_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'ellipse',
		'include_as_constraint': True,
		'search': 'segment',
		'number_of_points': 2,
	},
	'directory': 'ellipse_segment_2_linear',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'polyhedral',
		'search': 'none',
	},
	'directory': 'feasible_intersect_trust_linear',
	'increase-radius': False,
	'replacement-strategy-params': {
		'strategy': 'adaptive-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'simplex',
		'search': 'simplex-search',
	},
	'directory': 'simplex_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}, {
	'trust_region_options': {
		'shape': 'circumscribed-ellipse',
		'search': 'none',
	},
	'directory': 'circumscribed_ellipse_linear',
	'increase-radius': True,
	'replacement-strategy-params': {
		'strategy': 'fixed-xsi',
	},
	'basis': 'linear',
}]
