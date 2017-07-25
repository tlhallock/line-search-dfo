
from octave.pydfox import dfoxs
from numpy import random
from oct2py import octave
import os
from octave.pyfovec import dfovec
from numpy import asarray

def getOctavePath():
	curfile = os.path.realpath(__file__)
	curdir = os.path.dirname(curfile)
	octavedir = os.path.join(curdir, '../octave')
	return octavedir

def addOctaveToPath():
	octave.addpath(getOctavePath())


def arraysMatch(expected, actual, tol=.0001):
	if len(expected) != len(actual):
		return False
	for i in range(len(expected)):
		if abs(expected[i] - actual[i]) > tol:
			return False
	return True

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