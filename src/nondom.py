
def dominates(v1, v2):
	n = v1.size;
	
	dominates = True
	
	i = 0
	while i < n and dominates:
		dominates = dominates and v1[i] <= v2[i]
		i = i+1
	
	return dominates


class NonDominatedSet:
	"""Contains a set of points that are non-dominated."""
	
	def __init__(self):
		self.frontier = []
	
	def dominates(self, otherVec):
		return any(dominates(v, otherVec) for otherVec in frontier)
	
	def remove_dominated(self, newVec):
		newFrontier = [value for value in frontier if not dominates(newVec, value)]
		frontier = newFrontier
	
	def add(self, vec):
		remove_dominated(self, vec)
		frontier.insert(vec)