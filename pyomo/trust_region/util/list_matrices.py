

def vector_to_matrix(vector):
	return [[v] for v in vector]


def transpose(X):
	return list(map(list, zip(*X)))


def sum(X, Y, inner, i, j):
	s = 0
	for k in range(inner):
		s += X[i][k] * Y[k][j]
	return s


def multiply(X, Y):
	n = len(X)
	m = len(Y[0])
	inner = len(X[0])
	if inner != len(Y):
		raise Exception('bad multiplication')
	return [
		[sum(X, Y, inner, i, j) for j in range(m)]
		for i in range(n)
	]


def determinant(matrix):
	if len(matrix) == 1:
		return matrix[0][0]

	ret = 0
	sgn = 1
	for j in range(len(matrix[0])):
		ret = ret + sgn * matrix[0][j] * determinant(
			[
				[
					matrix[ii][jj]
					for jj in range(len(matrix))
					if jj != j
				]
				for ii in range(1, len(matrix))
			]
		)
		sgn = -sgn

	return ret
