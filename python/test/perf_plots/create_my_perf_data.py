

import os
from program import Program
from numpy import array_equal
from numpy import asarray
from numpy import zeros
from utilities.sys_utils import createObject
from algorithms import trust_filter
from utilities import functions
from octave.pyfovec import dfovec
from octave.pydfox import dfoxs


def runMyAlgorithm(dfovecProgram):
	constants = trust_filter.Constants()
	return trust_filter.trust_filter(dfovecProgram, constants, plot=True)