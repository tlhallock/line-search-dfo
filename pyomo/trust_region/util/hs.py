import numpy as np

OBJECTIVES = [
	{
		'problem': 21,
		'objective': lambda x: 0.01 * x[0] ** 2 + x[1] ** 2 - 100,
		'constraints': np.array([
			[-10, 1, -10],
		]),
		'bounds': np.array([
			[2, 50],
			[-50, 50]
		]),
		'minimizer': np.array([2, 0]),
		'minimum': -99.96,
		'dimension': 2,
		'x0': np.array([-1, -1]),
		'feasible': False,
		'f0': -98.99
	}, {
		'problem': 25,
		'objective': lambda x: sum([
			(-0.01*i + np.exp(-1.0/x[0] * (25 + (-50 * np.log(0.01 * i)) ** (2.0/3) - x[1]) ** x[2])) ** 2
			for i in range(1, 100)
		]),
		'constraints': np.zeros((0, 4)),
		'bounds': np.array([
			[0.1, 100],
			[0, 25.6],
			[0, 5]
		]),
		'minimizer': np.array([50, 25, 1.5]),
		'minimum': 0,
		'dimension': 3,
		'x0': np.array([100, 12.5, 3]),
		'feasible': True,
		'f0': 32.835
	}, {
		'problem': 44,
		'objective': lambda x: x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3],
		'constraints': np.array([
			[1, 2, 0, 0, 8],
			[4, 1, 0, 0, 12],
			[3, 4, 0, 0, 12],
			[0, 0, 2, 1, 8],
			[0, 0, 1, 2, 8],
			[0, 0, 1, 1, 5],
		]),
		'bounds': np.array([
			[0, np.inf],
			[0, np.inf],
			[0, np.inf]
		]),
		'minimizer': np.array([0, 3, 0, 4]),
		'minimum': -15,
		'dimension': 4,
		'x0': np.array([0, 0, 0, 0]),
		'feasible': True,
		'f0': 0
	}, {
		'problem': 76,
		'objective': lambda x: x[0] ** 2 + 0.5 * x[1] ** 2 + x[2] ** 2 + 0.5 * x[3] ** 2 - x[0] * x[2] + x[2] * x[3] - x[0] - 3 * x[1] + x[2] - x[3],
		'constraints': np.array([
			[1, 2, 1, 1, 5],
			[3, 1, 2, -1, 4],
			[0, -1, 0, -4, -1.5],
		]),
		'bounds': np.array([
			[0, np.inf],
			[0, np.inf],
			[0, np.inf],
			[0, np.inf]
		]),
		'minimizer': np.array([0.2727273, 2.090909, -2.6e-9, 0.5454545]),
		'minimum': -4.681818181,
		'dimension': 4,
		'x0': np.array([0.5, 0.5, 0.5, 0.5]),
		'feasible': True,
		'f0': -1.25
	}
# 231
# 224
# 250
# 253
# 331
# 24
# 35
# 37
# 45
# 224
# 232
# 251
# 268
# 340
]


# for p in OBJECTIVES:
# 	print(p['problem'])
# 	print(p['objective'](p['x0']), p['f0'])


# 21, 24, 25, 35, 36, 37, 44, 45, 76, 224, 231, 232, 250, 251