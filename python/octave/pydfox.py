from numpy import zeros
from numpy import ones
from numpy import cos
from numpy import sin
from numpy import log
from numpy import asarray
from numpy import sqrt

# This is a translation of dfoxs to python
# This is not tested at all...


def dfoxs(n, nprob, factor):
	x = zeros(n, dtype=float)

	if nprob == 1:  #     linear function - full rank or rank 1.
		x = ones(n)
	elif nprob == 2:  #     linear function - full rank or rank 1.
		x = ones(n)
	elif nprob == 3:  #     linear function - full rank or rank 1.
		x = ones(n)
	elif nprob == 4:  #     rosenbrock function.
		x[1 - 1] = -1.2
		x[2 - 1] = 1
	elif nprob == 5:  #     helical valley function.
		x[1 - 1] = -1
	elif nprob == 6:  #     powell singular function.
		x[1 - 1] = 3
		x[2 - 1] = -1
		x[3 - 1] = 0
		x[4 - 1] = 1
	elif nprob == 7:  #     freudenstein and roth function.
		x[1 - 1] = .5
		x[2 - 1] = -2
	elif nprob == 8:  #     bard function.
		x[1:3] = 1
	elif nprob == 9:  #     kowalik and osborne function.
		x[1 - 1] = .25
		x[2 - 1] = .39
		x[3 - 1] = .415
		x[4 - 1] = .39
	elif nprob == 10:  #     meyer function.
		x[1 - 1] = .02
		x[2 - 1] = 4000
		x[3 - 1] = 250
	elif nprob == 11:  #     watson function.
		x = .5 * ones(n)
	elif nprob == 12:  #     box 3-dimensional function.
		x[1 - 1] = 0
		x[2 - 1] = 10
		x[3 - 1] = 20
	elif nprob == 13:  #     jennrich and sampson function.
		x[1 - 1] = .3
		x[2 - 1] = .4
	elif nprob == 14:  #     brown and dennis function.
		x[1 - 1] = 25
		x[2 - 1] = 5
		x[3 - 1] = -5
		x[4 - 1] = -1
	elif nprob == 15:  #     chebyquad function.
		for k in range(n):
			x[k] = (k + 1)/(n+1)
	elif nprob == 16:  #     brown almost-linear function.
		x = .5*ones(n,1)
	elif nprob == 17:  #     osborne 1 function.
		x[1 - 1] = .5
		x[2 - 1] = 1.5
		x[3 - 1] = 1
		x[4 - 1] = .01
		x[5 - 1] = .02
	elif nprob == 18:  #     osborne 2 function.
		x[1 - 1] = 1.3
		x[2 - 1] = .65
		x[3 - 1] = .65
		x[4 - 1] = .7
		x[5 - 1] = .6
		x[6 - 1] = 3
		x[7 - 1] = 5
		x[8 - 1] = 7
		x[9 - 1] = 2
		x[10 - 1] = 4.5
		x[11 - 1] = 5.5
	elif nprob == 19:  # bdqrtic
		x = ones(n)
	elif nprob == 20:  # cube
		x = .5 * ones(n)
	elif nprob == 21:  # mancino
		for i in range(n):
			ss = 0
			for j in range(n):
				ss = ss+(sqrt((i + 1)/(j + 1))*((sin(log(sqrt((i + 1)/(j + 1))))) ** 5+(cos(log(sqrt((i + 1)/(j + 1))))) ** 5))
			x[i] = -8.710996e-4 * ((i-50 + 1) ** 3 + ss)
	elif nprob == 22:  # Heart8
		x = asarray([-.3, -.39, .3, -.344, -1.2, 2.69, 1.59, -1.5])
	return factor * x
