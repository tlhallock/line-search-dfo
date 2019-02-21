import numpy


class Objective:
	def __init__(self, minor_speed=1e-1, amplitude=0.5, freq=2):
		self.minor_speed = minor_speed
		self.amplitude = amplitude
		self.freq = freq

	def evaluate(self, x):
		return 0.9 * (x[0]) + 0.1 * (self.minor_speed * x[0] + (x[1] - self.amplitude * x[0] * numpy.sin(self.freq * x[0])) ** 2)


class Objective2:
	def __init__(self, rotation_angle=numpy.pi, a=1, b=100):
		self.rotationMatrix = numpy.array([
			[numpy.cos(rotation_angle), -numpy.sin(rotation_angle)],
			[numpy.sin(rotation_angle), numpy.cos(rotation_angle)]
		])
		self.a = a
		self.b = b

	def evaluate(self, x):
		rotated = numpy.dot(self.rotationMatrix, x)
		return (self.a - rotated[0]) ** 2 + self.b * (rotated[0] - rotated[1] ** 2) ** 2


class RandomOrder2:
	def __init__(self, dim):
		self.evaluations = [(
			numpy.random.normal(numpy.zeros(dim), 1)
		)]

	def evaluate(self, x):
		pass
















