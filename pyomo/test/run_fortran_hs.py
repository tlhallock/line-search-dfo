import traceback
import numpy
import os
import json
import numpy as np
import time
import sys

from trust_region.algorithm.always_feasible_algorithm import AlgorithmParams
from trust_region.algorithm.always_feasible_algorithm import always_feasible_algorithm
from trust_region.algorithm.always_feasible_algorithm import restart_always_feasible_algorithm
from trust_region.util.hs_fortran_caller import HsProblem
from trust_region.util.utils import write_json
from trust_region.util.plots import create_plot_on

numpy.set_printoptions(linewidth=255)



RESTART = json.loads('''
{
  "iteration": 35,
  "number-of-evaluations": 89,
  "previous-sample-points": [
    [
      21.85080234742815,
      12.619314573095094,
      12.44527711963887
    ],
    [
      21.863491088923837,
      12.61828353002091,
      12.44488021604705
    ],
    [
      21.85336924458881,
      12.624266157307815,
      12.446767710461634
    ],
    [
      21.854334976205156,
      12.61627681397466,
      12.454213301351134
    ],
    [
      21.85080234742815,
      12.619314573095094,
      12.454430916709397
    ],
    [
      21.857251229680365,
      12.622950662272563,
      12.448081307862136
    ],
    [
      21.859620906945278,
      12.620988475220084,
      12.448081307862136
    ],
    [
      21.858975932821714,
      12.620446751854969,
      12.447043437881165
    ],
    [
      21.853326276645458,
      12.619628957038076,
      12.450802723053048
    ],
    [
      21.856077864926867,
      12.623983175174795,
      12.448001252109309
    ]
  ],
  "previous-sample-values": [
    -3431.6874516063003,
    -3433.2901878636553,
    -3433.8485018053716,
    -3433.879921461462,
    -3434.2115392774303,
    -3434.4630254457697,
    -3434.301445044413,
    -3433.766393622346,
    -3433.6931958112405,
    -3434.5374765166766
  ],
  "outer-trust-region": {
    "type": "L1",
    "center": [
      21.86383896053765,
      12.619811453306106,
      12.448269059849387
    ],
    "radius": 0.0023399414016403034
  },
  "center": [
    21.86383896053765,
    12.619811453306106,
    12.448269059849387
  ],
  "objective-value": -3434.6955936036975,
  "radius": 0.0023399414016403034,
  "tr-contained-in-constraints": false,
  "inner-trust-region": {
    "center": [
      21.862961484155957,
      12.618933976924412,
      12.447391583467693
    ],
    "volume": 9.826669150353835e-09,
    "ds": [
      [
        0.0032174177833326208,
        0.0,
        0.0
      ],
      [
        -0.0014624650199479563,
        0.0,
        0.0
      ],
      [
        0.0,
        0.003217417783334398,
        0.0
      ],
      [
        0.0,
        -0.0014624650199461795,
        0.0
      ],
      [
        0.0,
        0.0,
        0.003217417783334398
      ],
      [
        0.0,
        0.0,
        -0.0014624650199461795
      ],
      [
        0.0004874883399814678,
        0.0009749766799629356,
        0.0009749766799629356
      ],
      [
        -7.999512511660019,
        -15.999025023320039,
        -15.999025023320039
      ],
      [
        -21.862961484155953,
        0.0,
        0.0
      ],
      [
        0.0,
        -12.618933976924412,
        0.0
      ],
      [
        0.0,
        0.0,
        -12.447391583467693
      ],
      [
        20.137038515844043,
        0.0,
        0.0
      ],
      [
        0.0,
        29.381066023075583,
        0.0
      ],
      [
        0.0,
        0.0,
        29.552608416532305
      ]
    ],
    "lambdas": [
      3.519990288001366,
      1.5999982014851495,
      3.5199902880033096,
      1.5999982014832057,
      3.5199902880033096,
      1.5999982014832057,
      1.5999982014812626,
      26255.40875884223,
      23918.998797683005,
      13805.644164027526,
      13617.969575394305,
      22030.766527143478,
      32144.121160798953,
      32331.795749432178
    ],
    "q": 3,
    "q^-1": [
      [
        2.1388039345609834e-06,
        0.0,
        0.0
      ],
      [
        0.0,
        2.1388039345609834e-06,
        0.0
      ],
      [
        0.0,
        0.0,
        2.1388039345609834e-06
      ]
    ],
    "l": [
      [
        683.7770383308148,
        0.0,
        0.0
      ],
      [
        0.0,
        683.7770383308148,
        0.0
      ],
      [
        0.0,
        0.0,
        683.7770383308148
      ]
    ],
    "l^-1": [
      [
        0.0014624650199444031,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0014624650199444031,
        0.0
      ],
      [
        0.0,
        0.0,
        0.0014624650199444031
      ]
    ]
  },
  "new-sample-points": [
    [
      21.862961484155957,
      12.618933976924412,
      12.447391583467693
    ],
    [
      21.865029722021657,
      12.618933976924412,
      12.447391583467693
    ],
    [
      21.862961484155957,
      12.621002214790114,
      12.447391583467693
    ],
    [
      21.862961484155957,
      12.618933976924412,
      12.449459821333395
    ],
    [
      21.853326276645458,
      12.619628957038076,
      12.450802723053048
    ],
    [
      21.856077864926867,
      12.623983175174795,
      12.448001252109309
    ],
    [
      21.858975932821714,
      12.620446751854969,
      12.447043437881165
    ],
    [
      21.859620906945278,
      12.620988475220084,
      12.448081307862136
    ],
    [
      21.863491088923837,
      12.61828353002091,
      12.44488021604705
    ],
    [
      21.857251229680365,
      12.622950662272563,
      12.448081307862136
    ]
  ],
  "new-sample-values": [
    -3434.0768515725144,
    -3434.401715511158,
    -3434.6396952953787,
    -3434.6474520658408,
    -3433.6931958112405,
    -3434.5374765166766,
    -3433.766393622346,
    -3434.301445044413,
    -3433.2901878636553,
    -3434.4630254457697
  ],
  "coefficients": [
    -3434.0768515725144,
    -0.3248639511366491,
    -0.5628437625418883,
    -0.5706004777021008,
    2.4985638447105885e-08,
    -0.00010640517575666308,
    -0.00010788612416945398,
    7.936614565551281e-08,
    -0.00018682400695979595,
    -3.1257513910532e-08
  ],
  "gradient": [
    -0.3249093984493084,
    -0.562905931986391,
    -0.5706630082357647
  ],
  "projected-gradient": [
    21.866178911938405,
    12.61747150191234,
    12.4494390471184
  ],
  "xsi": 0.0035099309527030615,
  "critical": false,
  "trial-point": [
    21.864814481097593,
    12.619424237018073,
    12.448168535779802
  ],
  "new-function-value": -3434.7157139019223,
  "rho": 1.0000013327850301,
  "iteration-result": "accepted"
}
''')


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

		if RESTART is not None:
			result = restart_always_feasible_algorithm(params, RESTART)
		else:
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
	# 21,
	# 224,
	# 231,
	# 232,
	# 24,
	# 25,
	# 35,
	# 36,
	37,
	# 44,
	# 45,
	# 76,
	# 250,
	# 251
]
for objective in OBJECTIVES:
	for tr_strategy in [{
	# 	'name': 'ellipse',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'none',
	# 	},
	# 	'requires-interior-x0': True,
	# 	'buffer-factor': 0.75
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
	# 	'name': 'ellipse_segment_2',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 2,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
	# 	'name': 'ellipse_segment_3',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 3,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
	# 	'name': 'ellipse_segment_4',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 4,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
	# 	'name': 'ellipse_segment_5',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 5,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
	# 	'name': 'polyhedral',
	# 	'params': {
	# 		'shape': 'polyhedral',
	# 		'search': 'anywhere',
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
	# 	'name': 'circumscribed_ellipse',
	# 	'params': {
	# 		'shape': 'circumscribed-ellipse',
	# 		'search': 'none',
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': None
	# }, {
		'name': 'ellipse_segment_1_buffered',
		'params': {
			'shape': 'ellipse',
			'search': 'segment',
			'number_of_points': 1,
		},
		'requires-interior-x0': False,
		'buffer-factor': 0.75
	# }, {
	# 	'name': 'ellipse_segment_2_buffered',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 2,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': .75
	# }, {
	# 	'name': 'ellipse_segment_3_buffered',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 3,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': .75
	# }, {
	# 	'name': 'ellipse_segment_4_buffered',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 4,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': .75
	# }, {
	# 	'name': 'ellipse_segment_5_buffered',
	# 	'params': {
	# 		'shape': 'ellipse',
	# 		'search': 'segment',
	# 		'number_of_points': 5,
	# 	},
	# 	'requires-interior-x0': False,
	# 	'buffer-factor': .75
	}]:
		run_on_objective(objective, tr_strategy)
