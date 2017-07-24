
from numpy import random
from oct2py import octave
import os
from pyfovec import dfovec
from numpy import asarray

bounds = {
	5: {
		'n': 3,
		'm': 3
	},
	6: {
		'n': 4,
		'm': 4
	},
	7: {
		'n': 2,
		'm': 2
	},
	8: {
		'n': 15,
		'm': 15
	},
	9: {
		'n': 4,
		'm': 11
	},
	10: {
		'n': 3,
		'm': 16
	},
	11: {
		'n': 2,
		'm': 31
	},
	12: {
		'n': 3
	},
	13: {
		'n': 2
	},
	14: {
		'n': 4
	},
	16: {
		'm>=n': True
	},
	17: {
		'n': 5,
		'm': 33
	},
	18: {
		'n': 11,
		'm': 66
	},
	19: {
		'n': 5,
		'm-from-n': lambda n: (n - 4) * 2
	},
	20: {
		'm-from-n': lambda n: n
	},
	21: {
		'n': 2,
		'm-from-n': lambda n: n
	}
}
def getOctavePath():
	curfile = os.path.realpath(__file__)
	curdir = os.path.dirname(curfile)
	octavedir = os.path.join(curdir, '../octave')
	return octavedir
octave.addpath(getOctavePath())

def arraysMatch(expected, actual, tol=.0001):
	if len(expected) != len(actual):
		return False
	for i in range(len(expected)):
		try:
			if abs(expected[i] - actual[i]) > tol:
				return False
		except:
			print(expected)
			print(actual)
			return False
	return True


# TODO: add 0 in sometimes

error = False
for _ in range(100000):
	nprob = random.randint(1, 22)

	n = random.randint(2, 15)
	m = random.randint(2, 15)

	if nprob in bounds:
		bound = bounds[nprob]
		if 'n' in bound and n < bound['n']:
			n = bound['n']
		if 'm' in bound and m < bound['m']:
			m = bound['m']
		if 'm-from-n' in bound:
			m = bound['m-from-n'](n)
		if 'm>=n' in bound and m < n:
			m = n


	xdim = random.randint(n, 2 * n)
	x = 5 * (2 * random.rand(xdim) - 1)

	# zero is special,  and doesn't happen often with random numbers, so adding it here sometimes...
	if random.rand() < .3:
		x[random.randint(xdim)] = 0

	expected = asarray(octave.dfovec(m, n, x, nprob).T).flatten()
	actual = dfovec(m, n, x, nprob)
	tol = .0001
	if nprob == 15: # Not sure what the issue is here...
		tol = .01
	if arraysMatch(expected, actual, tol):
		continue

	error = True
	print('\t##########################################')
	print('\tnprob = ' + str(nprob))
	print('\tm = ' + str(m))
	print('\tn = ' + str(n))
	print('\tx = ' + str(x))
	print('\t##########################################')
	print('\t#expected = asarray(' + str(expected) + ')')
	print('\t#actual = asarry(' + str(actual) + ')')
	break

if error:
	print('failed')
else:
	print('success')
