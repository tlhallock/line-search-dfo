
import numpy as np

# https://en.wikipedia.org/wiki/Test_functions_for_optimization

WIKIPEDIA_OBJECTIVES = [{
	'name': 'Rastrigin',
	'func': lambda x: 10 * 2 + sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x]),
	'min': np.array([0, 0]),
	'value': 0,
	'lb': np.array([-5.12, -5.12]),
	'ub': np.array([5.12, 5.12])
}, {
	'name': 'Ackley',
	'func': lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) -
					np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20,
	'min': np.array([0, 0]),
	'value': 0,
	'lb': np.array([-5, -5]),
	'ub': np.array([5, 5])
}, {
	'name': 'Sphere',
	'func': lambda x: sum(xi * xi for xi in x),
	'min': np.array([0, 0]),
	'value': 0,
	'lb': None,
	'ub': None
}, {
	'name': 'Rosenbrock',
	'func': lambda x: sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)),
	'min': np.array([1, 1]),
	'value': 0,
	'lb': None,
	'ub': None
}, {
	'name': 'Beale',
	'func': lambda x: (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2,
	'min': np.array([3, 0.5]),
	'value': 0,
	'lb': np.array([-4.5, -4.5]),
	'ub': np.array([4.5, 4.5])
}, {
	'name': 'Goldstein-Price',
	'func': lambda x: (
		1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
	) * (
		30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
	),
	'min': np.array([0, -1]),
	'value': 3,
	'lb': np.array([-2, -2]),
	'ub': np.array([2, 2])
}, {
	'name': 'Booth',
	'func': lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2,
	'min': np.array([1, 3]),
	'value': 0,
	'lb': np.array([-10, -10]),
	'ub': np.array([10, 10])
}, {
	'name': 'Bukin-N.6',
	'func': lambda x: 100 * np.sqrt(np.abs(x[1] - 0.01*x[0])) + 0.01 * np.abs(x[0] + 10),
	'min': np.array([-10, 1]),
	'value': 0,
	'lb': np.array([-15, -3]),
	'ub': np.array([-5, 3])
}, {
	'name': 'Matyas',
	'func': lambda x: 0.26 * (x[0]**2 + x[1] ** 2) - 0.48 * x[0] * x[1],
	'min': np.array([0, 0]),
	'value': 0,
	'lb': np.array([-10, -10]),
	'ub': np.array([10, 10])
}, {
	'name': 'Levi',
	'func': lambda x: np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2) + (
				x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2),
	'min': np.array([1, 1]),
	'value': 0,
	'lb': np.array([-10, -10]),
	'ub': np.array([10, 10])
}, {
	'name': 'Himelblaus',
	'func': lambda x: (x[0] + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
	'min': np.array([3, 2]),
	'value': 0,
	'lb': np.array([-5, -5]),
	'ub': np.array([5, 5])
}, {
	'name': 'Three-hump',
	'func': lambda x: 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2,
	'min': np.array([0, 0]),
	'value': 0,
	'lb': np.array([-5, -5]),
	'ub': np.array([5, 5])
}, {
	'name': 'Easom',
	'func': lambda x: -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2)),
	'min': np.array([np.pi, np.pi]),
	'value': -1,
	'lb': np.array([-100, -100]),
	'ub': np.array([100, 100])
}, {
	'name': 'Cross-in-tray',
	'func': lambda x: -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(
		np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)
	)) + 1) ** 0.1,
	'min': np.array([1.34941, -1.34941]),
	'value': -2.06261,
	'lb': np.array([-10, -10]),
	'ub': np.array([10, 10])
}, {
	'name': 'Eggholder',
	'func': lambda x: -(x[1] + 47) * np.sin(np.sqrt(x[0]/2 + x[1] + 47)) - x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47))),
	'min': np.array([512, 404.2319]),
	'value': -959.6407,
	'lb': np.array([-512, -512]),
	'ub': np.array([512, 512])
}, {
	'name': 'Holder-table',
	'func': lambda x: -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))),
	'min': np.array([8.05502, 9.66459]),
	'value': -19.2085,
	'lb': np.array([-10, -10]),
	'ub': np.array([10, 10])
}, {
	'name': 'McCormick',
	'func': lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1,
	'min': np.array([-0.54719, -1.54719]),
	'value': -1.9133,
	'lb': np.array([-1.5, -3]),
	'ub': np.array([4, 4])
}, {
	'name': 'Schaffer-N.2',
	'func': lambda x: 0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) ** 2 / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2,
	'min': np.array([0, 0]),
	'value': 0,
	'lb': np.array([-100, -100]),
	'ub': np.array([100, 100])
}, {
	'name': 'Schaffer-N.4',
	'func': lambda x: 0.5 + (np.cos(np.sin(np.abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5) ** 2 / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2,
	'min': np.array([0, 1.25313]),
	'value': 0.292579,
	'lb': np.array([-100, -100]),
	'ub': np.array([100, 100])
}, {
	'name': 'Styblinski-Tang',
	'func': lambda x: 0.5 * sum(xi ** 4 - 16 * xi * 2 + 5 * xi for xi in x),
	'value': -39.166165,
	'min': np.array([-2.903534, -2.903534]),
	'lb': np.array([-5, -5]),
	'ub': np.array([5, 5])
}]



