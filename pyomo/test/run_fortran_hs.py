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
  "iteration": 54,
  "number-of-evaluations": 111,
  "previous-sample-points": [
    [
      19.991643004381373,
      10.995312896137003,
      15.005685818845413
    ],
    [
      19.979020487774918,
      10.99607194766959,
      15.010003502688832
    ],
    [
      19.991155559925875,
      10.988719632139711,
      15.010704957355845
    ],
    [
      19.991643004381373,
      10.995312896137003,
      15.00843303713582
    ],
    [
      19.995656540337404,
      10.994604187621892,
      15.0049771103303
    ],
    [
      19.994481002872398,
      10.992493102927225,
      15.006450617869465
    ],
    [
      19.994481002872757,
      10.994811763540024,
      15.007127262930977
    ],
    [
      19.991643004381373,
      10.998591977421777,
      15.00389552117826
    ],
    [
      19.989728469966323,
      10.992769002955672,
      15.007597366236935
    ],
    [
      19.987629468425343,
      10.996021604652114,
      15.006394527360525
    ]
  ],
  "previous-sample-values": [
    -3298.4653768038647,
    -3297.558877888667,
    -3297.50968865402,
    -3299.0692548620104,
    -3298.759126471224,
    -3298.2556939159545,
    -3299.1001498455516,
    -3299.055412354069,
    -3297.806476535616,
    -3298.1714993029295
  ],
  "outer-trust-region": {
    "type": "L1",
    "center": [
      19.99643913128277,
      10.998285866511083,
      15.003383175055708
    ],
    "radius": 0.00968843562870246
  },
  "center": [
    19.99643913128277,
    10.998285866511083,
    15.003383175055708
  ],
  "objective-value": -3299.642358203215,
  "radius": 0.00968843562870246,
  "tr-contained-in-constraints": false,
  "inner-trust-region": {
    "center": [
      19.994747543754915,
      10.994747543754915,
      14.999999999999998
    ],
    "volume": 1.9082498670935335e-07,
    "ds": [
      [
        0.011380023156558396,
        -0.0005507324270549488,
        -0.0008397107497500464
      ],
      [
        -0.00799684810084855,
        0.00038700479803786245,
        0.0005900725527549425
      ],
      [
        -0.0006401045628075268,
        0.013226758384869086,
        -0.0021088365780564046
      ],
      [
        0.0002976326622851121,
        -0.006150112872534307,
        0.0009805564301841793
      ],
      [
        -0.0006693210502861218,
        -0.0014462301870498682,
        0.013071610684411537
      ],
      [
        0.00032285567027140775,
        0.0006976078463498613,
        -0.0063052605729918545
      ],
      [
        -6.043752103792127,
        -13.058989750604525,
        -19.911255513131785
      ],
      [
        0.0013229788487426624,
        0.0028586161260910703,
        0.004358578817164885
      ],
      [
        -19.994747543754915,
        0.9676391419974358,
        1.4753752448520496
      ],
      [
        0.5320871422075218,
        -10.994747543754919,
        1.7529711447129959
      ],
      [
        0.7680626356371477,
        1.6595852897928185,
        -15.0
      ],
      [
        0.005252456245084856,
        -0.0002541908690395258,
        -0.0003875689803889097
      ],
      [
        -0.0002541908690395257,
        0.005252456245084858,
        -0.0008374366214284916
      ],
      [
        -1.3825127441468659,
        -2.987253521627073,
        27.000000000000004
      ]
    ],
    "lambdas": [
      7.992854227257368,
      5.616652995188868,
      9.289924125371071,
      4.319583097070169,
      6.370986185139632,
      3.0731276331647517,
      41747567.09779504,
      0.028277001931445365,
      28985024.986984763,
      8766280.31855332,
      11321510.159553954,
      1.3558869916164953,
      0.6526978341749726,
      36668826.12833774
    ],
    "q": 3,
    "q^-1": [
      [
        1.3794148957387305e-05,
        -6.675632404211108e-07,
        -1.0178446039888844e-06
      ],
      [
        -6.675632404211108e-07,
        1.3794148957387309e-05,
        -2.199299710333742e-06
      ],
      [
        -1.0178446039888844e-06,
        -2.199299710333742e-06,
        1.9878156222464776e-05
      ]
    ],
    "l": [
      [
        269.24803612887104,
        0.0,
        0.0
      ],
      [
        13.045454420438906,
        269.56388638,
        0.0
      ],
      [
        18.530958685840094,
        37.06191737168019,
        226.82994032096025
      ]
    ],
    "l^-1": [
      [
        0.003714047516845645,
        0.0,
        0.0
      ],
      [
        -0.00017974009147520947,
        0.0037096957364322744,
        0.0
      ],
      [
        -0.00027405266070837554,
        -0.0006061300226203984,
        0.004408589089187336
      ]
    ]
  },
  "new-sample-points": [
    [
      19.991643004381373,
      10.995312896137003,
      15.005685818845413
    ],
    [
      19.979020487774918,
      10.99607194766959,
      15.010003502688832
    ],
    [
      19.991155559925875,
      10.988719632139711,
      15.010704957355845
    ],
    [
      19.991643004381373,
      10.995312896137003,
      15.00843303713582
    ],
    [
      19.995656540337404,
      10.994604187621892,
      15.0049771103303
    ],
    [
      19.994481002872398,
      10.992493102927225,
      15.006450617869465
    ],
    [
      19.994481002872757,
      10.994811763540024,
      15.007127262930977
    ],
    [
      19.991643004381373,
      10.998591977421777,
      15.00389552117826
    ],
    [
      19.989728469966323,
      10.992769002955672,
      15.007597366236935
    ],
    [
      19.987629468425343,
      10.996021604652114,
      15.006394527360525
    ]
  ],
  "new-sample-values": [
    -3298.4653768038647,
    -3297.558877888667,
    -3297.50968865402,
    -3299.0692548620104,
    -3298.759126471224,
    -3298.2556939159545,
    -3299.1001498455516,
    -3299.055412354069,
    -3297.806476535616,
    -3298.1714993029295
  ],
  "coefficients": [
    -3297.558022109006,
    -0.7048019417561591,
    -1.3850333096052054,
    -1.3706151087535545,
    8.084307410172187e-05,
    -0.0006555848522111773,
    -0.0006572018610313535,
    0.0001797707736841403,
    -0.0013082149962428957,
    -9.415089152753353e-07
  ],
  "gradient": [
    -0.7052181626526315,
    -1.3854449231919923,
    -1.3711729197479665
  ],
  "projected-gradient": [
    20.000000009976603,
    11.000000009963243,
    14.99999999004821
  ],
  "xsi": 0.00520231541159679,
  "critical": false,
  "trial-point": [
    19.99653445704759,
    10.998167523282472,
    15.00341929661239
  ],
  "new-function-value": -3299.6305272693953,
  "rho": 1.0000038275858203,
  "iteration-result": "step too small"
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
	# 37,
	# 44,
	# 45,
	# 76,
	250,
	# 251
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
