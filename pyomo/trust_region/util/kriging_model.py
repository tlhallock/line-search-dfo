import numpy as np
from scipy.linalg import lu
from trust_region.util.plots import create_plot_on

np.set_printoptions(linewidth=255)


def cov(x, y):
	#return np.dot(x - np.mean(x), y - np.mean(y)) / (len(x) - 1.0)
	# return 100 * np.exp(-0.3 * np.linalg.norm(x - y))
	#return 1.0 / np.linalg.norm(x-y) ** 2
	d = -0.3 * np.linalg.norm(x - y)
	return 100 * (1 + d + d ** 2 / 2.0)


class KrigingModel:
	def __init__(self, points, values):
		self.points = points
		self.values = values
		covariance = np.array([
			[cov(u, v) for v in points]
			for u in points
		])
		self.cinv = np.array(np.linalg.inv(np.bmat([
			[covariance, np.ones((covariance.shape[0], 1))],
			[np.ones((1, covariance.shape[1])), np.zeros((1, 1))]
		])))

	def evaluate(self, x0):
		c0 = np.concatenate([
			np.array([
				cov(u, x0)
				for u in self.points
			]),
			np.ones(1)
		])
		w = np.dot(self.cinv, c0)
		return np.dot(w[:-1], self.values), cov(self.values, self.values) - np.dot(w, c0)


def test():
	def func(x):
		return np.linalg.norm(x - np.array([2, 2])) ** 2

	points = np.random.random((25, 2))
	# points = np.array([
	# 	[x, y]
	# 	for x in np.linspace(0, 1, 4)
	# 	for y in np.linspace(0, 1, 4)
	# ])
	model = KrigingModel(points, np.array([func(u) for u in points]))
	#x0 = np.random.random(2)
	#x0 = points[0] + np.random.random(2) / 100

	for p in points:
		print(model.evaluate(p))
		print(func(p))

	p = create_plot_on('kriging.png', [-0, -0], [1, 1])
	p.add_points(points, label='sample points', color='b')
	p.add_contour(lambda x: model.evaluate(x)[0], label='model', color='r')
	p.add_contour(lambda x: model.evaluate(x)[1], label='model', color='y')
	p.add_contour(lambda x: func(x), label='actual', color='g')
	p.save()

