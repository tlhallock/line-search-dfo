import abc


class TrustRegionStrategy(metaclass=abc.ABCMeta):
	def __init__(self, context):
		self.context = context

	@abc.abstractmethod
	def find_trust_region(self):
		pass

	@abc.abstractmethod
	def add_to_plot(self, plot_object):
		pass
