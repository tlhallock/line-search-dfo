
import math
import numpy as np
import pdb

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%0.03g" % x))

class MultiIndex:
	def __init__(self, t):
		self.t = t
		
	def __repr__(self):
		return ','.join([str(ti) for ti in self.t])
	
	def as_exponent(self, x):
		return np.prod([math.pow(xi, ti) for xi, ti in zip(x, self.t)])

def get_max_idx(idx):
	if idx == 0:
		return 0
	if idx <= 2:
		return 2
	if idx <= 6:
		return 6

def vander(sigma, alphas):
	return np.array([
		[alpha.as_exponent(p) for alpha in alphas]
		for p in sigma
	])

def ensure_they_match(V, Phi, sigma, alphas):
	v1 = vander(sigma, alphas).T
	v2 = Phi @ v1
	v3 = v2 - V.T
	
	if np.max(np.abs(v3)) > 1e-4:
		pdb.set_trace()
		raise Exception()


def test_it(sigma, alphas):
	# Initialize
	V = vander(sigma, alphas)
	Phi = np.eye(len(sigma))
	
	h = len(V)
	for k in range(h):
		print('-------------------------------------------')
		print(sigma)
		print('-')
		print(V)
		print('-')
		print(Phi)
		print('-------------------------------------------')
		
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
















import numpy as np
n1 = 3
n2 = 2
n = n1 + n2
V = np.random.random((n, n))
A = np.block([[np.eye(n1), np.zeros((n1, n2))],	[np.random.random((n2, n1)), np.zeros((n2, n2))]])
phi = np.linalg.solve(V, A)

np.linalg.norm(V@phi - A)


def e(idx):
	r = np.zeros((n, 1))
	r[idx, 0] = 1
	return r


def rm_col(i, j):
	return np.eye(n) - e(i).T@A@e(j) * e(i)@e(j).T


rr = 3
rc = 0
R = rm_col(rr, rc)
phi2 = np.linalg.solve(V, R@A)
np.linalg.norm(V@phi2 - R@A)







phi2 = np.linalg.solve(V, (np.eye(n) - e(rr).T@A@e(rc) * e(rr)@e(rc).T)@A)
np.linalg.norm(V@phi2 - R@A)


phi2 = np.linalg.solve(V, A - e(rr).T@A@e(rc) * e(rr)@e(rc).T)
np.linalg.norm(V@phi2 - R@A)


phi2 = np.linalg.solve(V, A) - np.linalg.solve(V, e(rr).T@A@e(rc) * e(rr)@e(rc).T)
np.linalg.norm(V@phi2 - R@A)


phi2 = phi - np.linalg.solve(V, A[rr, rc] * e(rr)@e(rc).T)
np.linalg.norm(V@phi2 - R@A)


phi2 = phi - A[rr, rc] * np.linalg.solve(V, e(rr)@e(rc).T)
np.linalg.norm(V@phi2 - R@A)








def rm_col2(j):
	e1 = np.zeros((n, 1))
	e1[n1:n] = 1
	return np.eye(n) - e1.T@A@e(j) * e1@e(j).T


rm_col2(0)@A


R2 = np.eye(n) - rm_col2(3, 0) - rm_col2(4, 0)
R2@A


R2 = np.eye(n) - rm_col2(3, 0) - rm_col2(4, 0)
R2@A


R2 = (np.eye(n) - (e(3).T@A@e(0) * e(3)@e(0).T + e(4).T@A@e(0) * e(4)@e(0).T))@A
R2@A


(np.eye(n) - A[3, 0] * e(3)@e(0).T + A[4, 0] * e(4)@e(0).T))@A
R2@A

(np.eye(n) - A[3, 0] * e(3)@e(0).T - A[4, 0] * e(4)@e(0).T))@A
R2@A


(np.eye(n) - (A[3, 0] * e(3)@e(0).T + A[4, 0] * e(4)@e(0).T))@A



	
(np.eye(n) - (A[3, 0] * e(3) + A[4, 0] * e(4))@e(0).T)@A



def rm_col3(j):
	e0 = np.zeros(n)
	e0[n1:n] = A[n1:n, j]
	return e0[:, np.newaxis]@e(j).T

R2 = rm_col3(0) + rm_col3(1) + rm_col3(2)
(np.eye(n) - R2) @A
phi2 = phi - np.linalg.solve(V, R2)
np.linalg.norm(V@phi2 - (np.eye(n) - R2)@A)
np.linalg.norm(V@phi - A)


def rm_cols(A):
	e0 = np.zeros_like(A)
	e0[n1:n, 0:n1] = A[n1:n, 0:n1]
	return e0


R2 = rm_cols(A)
(np.eye(n) - R2) @A
phi2 = phi - np.linalg.solve(V, R2)
np.linalg.norm(V@phi2 - (np.eye(n) - R2)@A)




phi2 = np.linalg.solve(V, (np.eye(n) - e(rr).T@A@e(rc) * e(rr)@e(rc).T)@A)
np.linalg.norm(V@phi2 - R@A)



rm_col3(0)

np.zeros((n, 1))

e0 = np.zeros(n)
e0[n1:n] = A[n1:n, 0]
e0[:, np.newaxis]@e(0).T

A[n1:n, 0]

np.array

A[:, 0] && 





R@V@phi - R@A = 0
V@(phi2) - R@A = 0


V@(phi2) = R@V@phi










a = np.random.random((4, 4))
b = np.random.random((4, 4))
x = np.linalg.solve(a, b)
np.linalg.norm(a@x - b)

i = np.eye(4)
i[1] = i[1] + 0.5 * i[0]

x2 = np.linalg.solve(a, i@a@x)
np.linalg.norm(a@x2 - i@b)


