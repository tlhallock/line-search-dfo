
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy


class PlotObject:
	def __init__(self):
		self.ax = None
		self.fig = None
		self.filename = None
		self.x = None
		self.y = None
		self.X = None
		self.Y = None
		self.Z = None

	def save(self):
		print('saving to {}'.format(self.filename))
		self.fig.savefig(self.filename)
		plt.close()

	def add_contour(self, func, label, color='k', lvls=6):
		for i in range(0, len(self.x)):
			for j in range(0, len(self.y)):
				self.Z[i, j] = func(numpy.array(([self.x[i], self.y[j]])))
		CS = plt.contour(self.X, self.Y, self.Z, lvls, colors=color)
		plt.clabel(CS, fontsize=9, inline=1)

	def add_points(self, points, label, color='r', s=20, marker="x"):
		self.ax.scatter(points[:, 0], points[:, 1], s=s, c=color, marker=marker, label=label)

	def add_arrow(self, x1, x2, color="red", width=0.05):
		self.ax.add_patch(patches.Arrow(
			x=x1[0], y=x1[1],
			dx=x2[0] - x1[0], dy=x2[1] - x1[1],
			facecolor=color,
			edgecolor=color,
			width=width
		))

	def add_polyhedron(self, A, b, label, color='b', lvls=[0.0, -1]):
		for i in range(A.shape[0]):
			func = lambda x: numpy.dot(A[i], x) - b[i]
			self.add_contour(func, label + '_' + str(i), color=color, lvls=lvls)


def create_plot(title, filename, bounds):
	ret_val = PlotObject()
	ret_val.fig = plt.figure()
	plt.title(title)
	ax = ret_val.fig.add_subplot(111)

	plt.legend(loc='lower left')
	ret_val.fig.set_size_inches(15, 15)
	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'

	ret_val.ax = ax
	ret_val.filename = filename

	ret_val.x = numpy.linspace(bounds.lbX, bounds.ubX, num=100)
	ret_val.y = numpy.linspace(bounds.lbY, bounds.ubY, num=100)
	X, Y = numpy.meshgrid(ret_val.x, ret_val.y)
	ret_val.X = X
	ret_val.Y = Y
	ret_val.Z = numpy.zeros((len(ret_val.y), len(ret_val.x)))

	return ret_val





bounds = {
	'lbX': -10,  # model.currentSet[0, 0] - 2 * model.modelRadius,
	'ubX': 10,  # model.currentSet[0, 0] + 2 * model.modelRadius,
	'lbY': -10,  # model.currentSet[0, 1] - 2 * model.modelRadius,
	'ubY': 10,  # model.currentSet[0, 1] + 2 * model.modelRadius
}
