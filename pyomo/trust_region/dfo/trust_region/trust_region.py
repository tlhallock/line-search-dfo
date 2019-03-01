import abc


class TrustRegion(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def add_shifted_pyomo_constraints(self, model):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def shift(self, points):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def unshift(self, points):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def shift_row(self, point):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def unshift_row(self, point):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def add_to_plot(self, plot_object, detailed=True):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def multiply_radius(self, factor):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def recenter(self, new_center):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def add_unshifted_pyomo_constraints(self, model):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def shift_pyomo_model(self, model):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def sample_shifted_region(self, num_points):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def shift_polyhedron(self, polyhedron):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def contains(self, point):
		raise Exception("Not implemented")

	@abc.abstractmethod
	def to_json(self):
		raise Exception("Not implemented")
