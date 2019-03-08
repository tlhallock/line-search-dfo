
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy
from .bounds import Bounds

class NoPlot:
	def save(self):
		pass

	def add_contour(self, *args, **kwargs):
		pass

	def add_points(self, *args, **kwargs):
		pass

	def add_point(self, *args, **kwargs):
		pass

	def add_arrow(self, *args, **kwargs):
		pass

	def add_polyhedron(self, *args, **kwargs):
		pass

	@property
	def ax(self):
		class axis:
			def add_artist(self, *args, **kwargs):
				pass

			def text(self, *args, **kwargs):
				pass

			def transAxes(self, *args, **kwargs):
				pass
		return axis()


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
		self.css = []

	def save(self):
		for cs in self.css:
			plt.clabel(cs, fontsize=9, inline=1)
			#artists, labels = cs.legend_elements()
			#self.ax.legend()
		self.ax.legend()
		self.ax.grid(True)
		print('saving to {}'.format(self.filename))
		self.fig.savefig(self.filename)
		plt.close()

	def add_contour(self, func, label, color='k', lvls=None):
		for i in range(0, len(self.x)):
			for j in range(0, len(self.y)):
				self.Z[j, i] = func(numpy.array([self.x[i], self.y[j]]))
		if lvls is None:
			self.css.append(plt.contour(self.X, self.Y, self.Z, colors=color))
		else:
			self.css.append(plt.contour(self.X, self.Y, self.Z, levels=lvls, colors=color))

	def add_points(self, points, label, color='r', s=20, marker="x"):
		self.ax.scatter(points[:, 0], points[:, 1], s=s, c=color, marker=marker, label=label)

	def add_point(self, point, label, color='r', s=20, marker="x"):
		self.ax.scatter([point[0]], [point[1]], s=s, c=color, marker=marker, label=label)

	def add_arrow(self, x1, x2, color="red", width=0.05):
		if width is None:
			self.ax.add_patch(patches.Arrow(
				x=x1[0], y=x1[1],
				dx=x2[0] - x1[0], dy=x2[1] - x1[1],
				facecolor=color,
				edgecolor=color
			))
		else:
			self.ax.add_patch(patches.Arrow(
				x=x1[0], y=x1[1],
				dx=x2[0] - x1[0], dy=x2[1] - x1[1],
				facecolor=color,
				edgecolor=color,
				width=width
			))

	def add_polyhedron(self, polyhedron, label, color='b', lvls=[-0.1, 0.0]):
		for i in range(polyhedron.A.shape[0]):
			func = lambda x: numpy.dot(polyhedron.A[i], x) - polyhedron.b[i]
			self.add_contour(func, label + '_' + str(i), color=color, lvls=lvls)


def create_plot(title, filename, bounds):
	if len(bounds.lb) != 2:
		return NoPlot()

	ret_val = PlotObject()
	ret_val.fig = plt.figure()
	plt.title(title)
	plt.ylim(bounds.lb[1], bounds.ub[1])
	plt.xlim(bounds.lb[0], bounds.ub[0])
	ax = ret_val.fig.add_subplot(111)

	plt.legend(loc='lower left')
	PLOT_SIZE = 10
	scale_factor = min(3.0, (bounds.ub[1] - bounds.lb[1]) / (bounds.ub[0] - bounds.lb[0]))
	ret_val.fig.set_size_inches(PLOT_SIZE, scale_factor * PLOT_SIZE)
	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'

	ret_val.ax = ax
	ret_val.filename = filename

	ret_val.x = numpy.linspace(bounds.lb[0], bounds.ub[0], num=100)
	ret_val.y = numpy.linspace(bounds.lb[1], bounds.ub[1], num=100)
	ret_val.X, ret_val.Y = numpy.meshgrid(ret_val.x, ret_val.y)
	ret_val.Z = numpy.zeros((len(ret_val.y), len(ret_val.x)))

	return ret_val

'''
bounds = {
	'lbX': -10,  # model.currentSet[0, 0] - 2 * model.modelRadius,
	'ubX': 10,  # model.currentSet[0, 0] + 2 * model.modelRadius,
	'lbY': -10,  # model.currentSet[0, 1] - 2 * model.modelRadius,
	'ubY': 10,  # model.currentSet[0, 1] + 2 * model.modelRadius
}
'''


def create_plot_on(filename, lb, ub):
	bounds = Bounds()
	bounds.extend(numpy.array([xi for xi in lb]))
	bounds.extend(numpy.array([xi for xi in ub]))
	return create_plot('a plot', filename, bounds)

