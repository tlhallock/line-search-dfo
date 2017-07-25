
from octave.pydfox import dfoxs
from numpy import random
from oct2py import octave
from numpy import asarray

from octave.python_octave_utils import addOctaveToPath
from octave.python_octave_utils import bounds
from octave.python_octave_utils import arraysMatch


addOctaveToPath()


error = False
while not error:
	nprob = random.randint(1, 22)
	factor = 50.0 * (2 * random.rand() - 1)
	n = random.randint(2, 15)

	if nprob in bounds:
		bound = bounds[nprob]
		if 'n' in bound and n < bound['n']:
			n = bound['n']

	expected = asarray(octave.dfoxs(n, nprob, factor).T).flatten()
	actual = dfoxs(n, nprob, factor)
	tol = .0001
	if arraysMatch(expected, actual):
		continue

	error = True
	print('\t##########################################')
	print('\tnprob = ' + str(nprob))
	print('\tn = ' + str(n))
	print('\tfactor = ' + str(factor))
	print('\t##########################################')
	print('\t#expected = asarray(' + str(expected) + ')')
	print('\t#actual = asarry(' + str(actual) + ')')

if error:
	print('failed')
else:
	print('success')
