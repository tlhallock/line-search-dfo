

class NoPlotDetails:
	def add_to_plot(self, plot_object):
		pass


class ObjectiveValue:
	def __init__(self):
		self.objective = None
		self.success = False
		self.point = None
		self.hot_start = None
		self.trust_region = None

