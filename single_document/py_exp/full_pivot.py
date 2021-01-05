import math
import numpy as np
import itertools
import pdb

np.set_printoptions(
	edgeitems=30,
	linewidth=100000, 
    #formatter=dict(float=lambda x: "%0.03g" % x)
)


class MatrixPrinting:
	@staticmethod
	def format_mat(mat):
		return [
			['|'] + ["%0.03g " % e for e in row] + ['|']
			for row in mat
		]

	@staticmethod
	def append_mat(fm1, fm2):
		return [
			row1 + ["\t"] + row2
			for row1, row2 in zip(fm1, fm2)
		]

	@staticmethod
	def print_mat(mat):
		lengths = {}
		for row in mat:
			for j, e in enumerate(row):
				lengths[j] = len(e) if j not in lengths or len(e) > lengths[j] else lengths[j]
		for row in mat:
			print("".join([e.rjust(lengths[j], ' ') for j, e in enumerate(row)]))
		



def vandermonde(sigma, alphas):
	return np.array([
		[alpha.as_exponent(p) / alpha.factorial() for p in sigma]
		for alpha in alphas
	])

def swap_vector_rows(vector, idx1, idx2):
	if idx1 == idx2:
		return
	t = vector[idx1]
	vector[idx1] = vector[idx2]
	vector[idx2] = t

def swap_matrix_rows(mat, idx1, idx2):
	if idx1 == idx2:
		return
	t = mat[idx1].copy()
	mat[idx1] = mat[idx2]
	mat[idx2] = t
	
def swap_matrix_cols(mat, idx1, idx2):
	if idx1 == idx2:
		return
	t = mat[:, idx1].copy()
	mat[:, idx1] = mat[:, idx2]
	mat[:, idx2] = t

def subset_matrix(mat, iis, jjs):
	r = np.zeros_like(mat)
	r[iis, jjs] = mat[iis, jjs].copy()
	return r

def get_max_element(mat, iis, jjs):
	i_max = -1
	j_max = -1
	v_max = -1
	for i in iis:
		for j in jjs:
			v_cur = np.abs(mat[i, j])
			if v_cur > v_max:
				v_max = v_cur
				i_max = i
				j_max = j
	return i_max, j_max, v_max

class MultiIndex:
	def __init__(self, t):
		self.t = tuple(t)
	
	def __repr__(self):
		return '(' + (','.join([str(ti) for ti in self.t])) + ')'
	
	@property
	def n(self):
		return len(self.t)
	
	@property
	def d(self):
		return np.sum([ti for ti in self.t])
	
	def as_exponent(self, x):
		return np.prod([np.math.pow(xi, ti) for xi, ti in zip(x, self.t)])
	
	def factorial(self):
		return np.prod([np.math.factorial(ti) for ti in self.t])
	
	def plus(self, other):
		return MultiIndex([a1 + a2 for a1, a2 in zip(self.t, other.t)])
	
	def one_greater(self):
		return [
			self.plus(MultiIndex.e(idx, self.n))
			for idx in range(self.n)
		]
	
	@staticmethod
	def e(idx, n):
		return MultiIndex([1 if i == idx else 0 for i in range(n)])
	
	@staticmethod
	def to_index(stars_and_bars, n):
		t = [0 for _ in range(n)]
		idx = 0
		for c in stars_and_bars:
			if c == '|':
				idx += 1
			elif c == '*':
				t[idx] += 1
			else:
				raise Exception()
		return MultiIndex(t)
	
	@staticmethod
	def zero(n):
		return MultiIndex([0 for _ in range(n)])

class LuFactorization:
	def __init__(self):
		self.n = None
		self.d = None
		self.sigma = None
		self.V = None
		self.alphas = None
		self.phi = None
		self.permutation = None
	
	def print_to_output(self, k):
		print('-------------------------------------------')
		print(k)
		print('-')
		print(self.permutation)
		print('-')
		print(self.sigma)
		print('-')
		
		MatrixPrinting.print_mat(
			MatrixPrinting.append_mat(
				MatrixPrinting.append_mat(
					MatrixPrinting.format_mat(self.phi@self.P_rows.T),
					MatrixPrinting.format_mat(self.P_rows),
				),
				MatrixPrinting.append_mat(
					MatrixPrinting.format_mat(self.M),
					MatrixPrinting.format_mat(self.V)
				)
			)
		)
		print('-------------------------------------------')
	
	@property
	def M(self):
		return vandermonde(self.sigma, self.alphas)
	
	@property
	def P_rows(self):
		return np.eye(self.d, dtype=np.int32)[self.permutation, :]
	
	def replacable_basis(self, k):
		idx_in_alpha = {alpha.t: idx for idx, alpha in enumerate(self.alphas)}
		used_indices = set()
		greater_indices = set()
		for used_idx in self.permutation[0:k]:
			used_indices.add(used_idx)
			for greater in self.alphas[used_idx].one_greater():
				if greater.t not in idx_in_alpha:
					continue
				greater_indices.add(idx_in_alpha[greater.t])
		if len(greater_indices) == 0:
			greater_indices.add(0)
		print(self.permutation)
		print('used: ', {i: self.alphas[i] for i in used_indices}, ' greater: ', {i: self.alphas[i] for i in greater_indices})
		replacable_alphas = greater_indices - used_indices
		return [idx for idx in range(self.d) if self.permutation[idx] in replacable_alphas]
	
	def swap_points(self, idx1, idx2):
		swap_matrix_rows(self.sigma, idx1, idx2)
		swap_matrix_cols(self.V, idx1, idx2)
	
	def assign_basis(self, idx1, idx2):
		# prev_idxs = {pidx: idx for idx, pidx in enumerate(self.permutation)}
		# swap_vector_rows(self.permutation, prev_idxs[idx1], prev_idxs[idx2])
		
		
		swap_vector_rows(self.permutation, idx1, idx2)
		swap_matrix_rows(self.phi, idx1, idx2)
		swap_matrix_rows(self.V, idx1, idx2)
	
	def row_op(self, row_op):
		self.phi = row_op@self.phi
		self.V = row_op@self.V
	
	def col_op(self, col_op):
		self.phi = self.phi@col_op
		self.V = self.V@col_op
	
	def divide_polynomial(self, idx, c):
		self.phi[idx] /= c
		self.V[idx] /= c
	
	def replace_point(self, idx, p):
		self.sigma[idx] = p
		self.V[:, idx] = self.phi@vandermonde(p[np.newaxis, :], self.alphas).flatten()
	
	def assert_match(self):
		em = self.M - vandermonde(self.sigma, self.alphas)
		e = np.max(np.abs(em))
		if e > 1e-4:
			pdb.set_trace()
			print(e)
		
		# em = self.phi@self.P_rows@self.M - self.V
		em = self.phi@self.M - self.V
		e = np.max(np.abs(em))
		print('error', e)
		if e > 1e-4:
			pdb.set_trace()
			print(e)
	
	@staticmethod
	def create_factorization(sigma, alphas):
		if len(sigma) != len(alphas):
			raise Exception()
		lu = LuFactorization()
		lu.n = sigma.shape[1]
		lu.d = sigma.shape[0]
		lu.sigma = sigma.copy()
		lu.alphas = alphas
		lu.V = vandermonde(sigma, alphas)
		lu.phi = np.eye(lu.d)
		lu.permutation = np.array(range(lu.d), dtype=np.int32)
		return lu



def full_pivoting(lu):
	lu.assert_match()
	
	lu.print_to_output(0)
	max_k = np.random.choice(range(1,lu.d))
	for k in range(max_k):
		print('========================================================================================================================')
		replacable = lu.replacable_basis(k)
		print(replacable, lu.permutation[[i for i in replacable]])
		imax, jmax, vmax = get_max_element(lu.V, replacable, range(k, lu.d))
		print(imax, jmax, vmax, 'adding:', lu.alphas[imax], 'of', replacable)
		
		lu.swap_points(k, jmax)
		print('swapped points', k, jmax)
		lu.print_to_output(k)
		lu.assert_match()
		
		lu.assign_basis(k, imax)
		print('swapped basis', k, imax)
		lu.print_to_output(k)
		lu.assert_match()
		
		if np.random.random() < 0.3:
			lu.replace_point(k, np.random.random(lu.n))
		
		lu.divide_polynomial(k, lu.V[k, k])
		print('divided polynomial')
		lu.print_to_output(k)
		lu.assert_match()
		
		row_op = np.eye(lu.d) - subset_matrix(lu.V, np.array(range(lu.d)) != k, k)
		#row_op = np.eye(lu.d) - subset_matrix(lu.V, range(k+1,lu.d), k)
		#col_op = np.eye(lu.d) - subset_matrix(lu.V, k, range(k+1,lu.d))
		#print('result')
		#print(lu.V@row_op)
		#print(lu.V@col_op)
		lu.row_op(row_op)
		#lu.col_op(row_op)
		print('performed operation')
		lu.print_to_output(k)
		lu.assert_match()


def get_stars_and_bars(n, k):
	return [
		['|' if i in p else '*' for i in range(n+k-1)]
		for p in itertools.combinations(range(n+k-1), r=n-1)
	]

def test_full_pivoting(n, d):
	print([
		(sab, MultiIndex.to_index(sab, n))
		for k in range(d+1)
		for sab in get_stars_and_bars(n, k)
	])
	alphas = [
		MultiIndex.to_index(sab, n)
		for k in range(d+1)
		for sab in get_stars_and_bars(n, k)
	]
	sigma = 2 * np.random.random((len(alphas), n)) - 1
	print(sigma)
	print(alphas)

	lu = LuFactorization.create_factorization(sigma, alphas)
	full_pivoting(lu)

	
	
if __name__ == '__main__':
	test_full_pivoting(n=2, d=2)


#for k in range(d+1):
	#for p in itertools.product(range(n), repeat=k):
		#print(p)
		#print(MultiIndex.to_index(p, n))










'''
def ensure_they_match(V, Phi, sigma, alphas):
	v1 = vander(sigma, alphas).T
	v2 = Phi @ v1
	v3 = v2 - V.T
	
	if np.max(np.abs(v3)) > 1e-4:
		pdb.set_trace()
		raise Exception()


def vandermonde

class LuFactorization



def test_it(sigma, alphas):
	# Initialize
	V = vander(sigma, alphas)
	Phi = np.eye(len(sigma))
	
	h = len(V)
	for k in range(h):
		
		# Pivot
		i_max = -1
		j_max = -1
		v_max = -1
		for i in range(k, h):
			for j in range(k, h): # Shouldn't look at all columns....
				v_cur = np.max(V[i, j])
				if v_cur > v_max:
					v_max = v_cur
					i_max = i
					j_max = j
		if i_max != k or j_max != k:
			if i_max != k:
				t = sigma[k].copy()
				sigma[k] = sigma[i_max]
				sigma[i_max] = t
			
			if j_max != k:
				t = Phi[k].copy()
				Phi[k] = Phi[j_max]
				Phi[j_max] = t
			
			if i_max != k:
				t = V[k].copy()
				V[k] = V[i_max]
				V[i_max] = t
			
			if j_max != k:
				t = V[:, k].copy()
				V[:, k] = V[:, j_max]
				V[:, j_max] = t
		
		# Row reduce
		t = V[k, k]
		if np.abs(t) < 1e-4:
			pdb.set_trace()
		for i in range(h):
			V[i, k] = V[i, k] / t
		for j in range(h):
			Phi[k, j] = Phi[k, j] / t
		
		for j in range(h):
			if j == k:
				continue
			tj = V[k, j]
			for i in range(h):
				V[i, j] = V[i, j] - tj * V[i, k]
			for i in range(h):
				Phi[j, i] = Phi[j, i] - tj * Phi[k, i]
		for i in range(k+1, h):
			t = V[i, k]
			V[i] = V[i] - t * V[k]
			Phi[i, :] = Phi[i, :] - Phi[k, :] / t
			# Phi[k, :] = Phi[k, :] - Phi[i, :] / t
		
		ensure_they_match(V, Phi, sigma, alphas)
		
	
	print('-------------------------------------------')
	print(sigma)
	print('-')
	print(V)
	print('-')
	print(Phi)
	print('-------------------------------------------')
	



alphas = [
	MultiIndex([0, 0]),
	MultiIndex([1, 0]),
	MultiIndex([0, 1]),
	MultiIndex([2, 0]),
	MultiIndex([1, 1]), 
	MultiIndex([0, 2]),
]

sigma = 2 * np.random.random((6, 2))

test_it(sigma, alphas)

'''
