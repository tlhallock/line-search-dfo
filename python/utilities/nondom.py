

def dominates(x, y):
	return all([x[i] <= y[i] for i in range(len(x))])


class NonDomSet:
	def __init__(self):
		self.non_dominated = set()

	def is_dominated(self, x):
		for point in self.non_dominated:
			if dominates(point, x):
				return True
		return False

	def add(self, x):
		self.non_dominated = [item for item in self.non_dominated if not dominates(x, item)]
		self.non_dominated.append(x)

	def size(self):
		return len(self.non_dominated)





