
from numpy.linalg import norm

import itertools
from numpy import inf as infinity

# Could be so much faster!!!!!!!!!!!!
# Also should replace this with a standard library implementation
# or make it a splay tree

class Node:
	def __init__(self, key, val):
		self.key = key
		self.val = val
		self.left = None
		self.right = None
		self.depth = 0

def _createTree(list, small, large):
	if small > large:
		return None
	if small == large:
		return Node(list[small][0], list[small][1])

	mid = int((small + large) / 2)
	ret = Node(list[mid][0], list[mid][1])
	ret.left = _createTree(list, small, mid-1)
	ret.right = _createTree(list, mid+1, large)
	return ret


class Tree:
	def __init__(self):
		self.root = None
		self.size = 0
		self.balOn = 8

	def balance(self):
		self.root = _createTree([i for i in self.range()], int(0), int(self.size-1))

	def add(self, key, val):
		self.root = self._addNode(self.root, Node(key, val), 0)
		self.size += 1
		if self.size >= self.balOn:
			self.balOn *= 2
			self.balance()

	def _addNode(self, current, node, depth):
		if current is None:
			node.depth=depth
			return node
		if node.key < current.key:
			current.left = self._addNode(current.left, node, depth+1)
		else:
			current.right = self._addNode(current.right, node, depth+1)

		return current

	def print(self, output):
		self._printAt(output, self.root, 0)

	def _printAt(self, output, current, depth):
		if current is None:
			return

		self._printAt(output, current.left, depth+1)
		for i in range(depth):
			output.write('  ')
		output.write(str(current.key) + "->" + str(current.val) + "\n")
		self._printAt(output, current.right, depth+1)

	def range(self, minimum=None, maximum=None, tol=0):
		yield from self._rangeR(minimum, maximum, self.root, tol)

	def _rangeR(self, minimum, maximum, current, tol):
		if current is None:
			return
		if minimum is None or current.key + tol >= minimum:
			yield from self._rangeR(minimum, maximum, current.left, tol)
		if (minimum is None or current.key + tol >= minimum) and (maximum is None or current.key <= maximum + tol):
			yield current.key, current.val
		if maximum is None or current.key < maximum + tol:
			yield from self._rangeR(minimum, maximum, current.right, tol)



class EvaluationHistory:
	def __init__(self, dim):
		self.dim = dim
		self.all_keys=[]
		self.all_values = []
		self.indices = [Tree() for _ in itertools.repeat(None, dim)]

	def balance(self):
		for t in self.indices:
			t.balance()

	def add(self, x, y):
		index = len(self.all_keys)
		self.all_keys.append(x)
		self.all_values.append(y)

		for i in range(self.dim):
			self.indices[i].add(x[i], index)

	def size(self):
		return len(self.all_keys)

	def get(self, center, tol=1e-10):
		minNorm = infinity
		minX = None
		minY = None

		for x, y in self.getBox(center, 0, tol):
			n = norm(x - center)
			if n >= minNorm:
				continue
			minNorm = n
			minX = x
			minY = y

		return minX, minY


	def getBox(self, center, dist, tol=1e-10):
		idxs = set(y for _, y in self.indices[0].range(center[0] - dist - tol, center[0] + dist + tol))
		for i in range(1, self.dim):
			idxs = idxs & set(y for _, y in self.indices[i].range(center[i] - dist - tol, center[i] + dist + tol))

		for i in idxs:
			yield self.all_keys[i], self.all_values[i]




