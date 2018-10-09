import abc
import numpy


class TrustRegion(metaclass=abc.ABCMeta):
	def __init__(self):
		pass

	@abc.abstractmethod
	def shift(self, points):
		pass

	@abc.abstractmethod
	def unshift(self, points):
		pass


class CircularTrustRegion(TrustRegion):
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def shift(self, points):
		shifted = numpy.empty(points.shape)
		for i in range(0, points.shape[0]):
			shifted[i, :] = (points[i, :] - self.center) / self.radius
		return shifted

	def unshift(self, points):
		unshifted = numpy.empty(points.shape)
		for i in range(0, points.shape[0]):
			unshifted[i, :] = points[i, :] * self.radius + self.center
		return unshifted
