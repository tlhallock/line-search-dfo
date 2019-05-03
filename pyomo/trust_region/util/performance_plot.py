






class PerformancePlot:
	def __init__(self):
		self.alg_to_prob = {}

	@property
	def num_probs(self):
		return len(set(
			key
			for _, value in self.alg_to_prob
			for key in value
		))

	@property
	def num_algorithms(self):
		return len(self.alg_to_prob)

