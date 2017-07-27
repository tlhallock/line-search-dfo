from numpy import asarray
from numpy import zeros
from numpy import arctan
from numpy import exp
from numpy import sin
from numpy import cos
from numpy import log
from math import sqrt
from math import pi

# This is a translation of dfovec to python


def dfovec(m, n, x, nprob, context=None):
	c13 = 1.3e1
	c14 = 1.4e1
	c29 = 2.9e1
	c45 = 4.5e1
	v = asarray([4.0e0, 2.0e0, 1.0e0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
	y1 = asarray([1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1,
			1.34e0, 2.1e0, 4.39e0])
	y2 = asarray([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2,
			2.46e-2])
	y3 = asarray([3.478e4, 2.861e4, 2.365e4, 1.963e4, 1.637e4, 1.372e4, 1.154e4, 9.744e3, 8.261e3, 7.03e3, 6.005e3,
			5.147e3, 4.427e3, 3.82e3, 3.307e3, 2.872e3])
	y4 = asarray([8.44e-1, 9.08e-1, 9.32e-1, 9.36e-1, 9.25e-1, 9.08e-1, 8.81e-1, 8.5e-1, 8.18e-1, 7.84e-1, 7.51e-1,
			7.18e-1, 6.85e-1, 6.58e-1, 6.28e-1, 6.03e-1, 5.8e-1, 5.58e-1, 5.38e-1, 5.22e-1, 5.06e-1, 4.9e-1,
			4.78e-1, 4.67e-1, 4.57e-1, 4.48e-1, 4.38e-1, 4.31e-1, 4.24e-1, 4.2e-1, 4.14e-1, 4.11e-1,
			4.06e-1])
	y5 = asarray([1.366e0, 1.191e0, 1.112e0, 1.013e0, 9.91e-1, 8.85e-1, 8.31e-1, 8.47e-1, 7.86e-1, 7.25e-1, 7.46e-1,
			6.79e-1, 6.08e-1, 6.55e-1, 6.16e-1, 6.06e-1, 6.02e-1, 6.26e-1, 6.51e-1, 7.24e-1, 6.49e-1,
			6.49e-1, 6.94e-1, 6.44e-1, 6.24e-1, 6.61e-1, 6.12e-1, 5.58e-1, 5.33e-1, 4.95e-1, 5.0e-1,
			4.23e-1, 3.95e-1, 3.75e-1, 3.72e-1, 3.91e-1, 3.96e-1, 4.05e-1, 4.28e-1, 4.29e-1, 5.23e-1,
			5.62e-1, 6.07e-1, 6.53e-1, 6.72e-1, 7.08e-1, 6.33e-1, 6.68e-1, 6.45e-1, 6.32e-1, 5.91e-1,
			5.59e-1, 5.97e-1, 6.25e-1, 7.39e-1, 7.1e-1, 7.29e-1, 7.2e-1, 6.36e-1, 5.81e-1, 4.28e-1, 2.92e-1,
			1.62e-1, 9.8e-2, 5.4e-2])

	fvec = zeros(m)
	sum = 0

	if nprob == 1:  # Linear function - full rank.
		for j in range(n):
			sum += x[j]
		temp = 2 * sum / m + 1
		for i in range(m):
			fvec[i] = -temp
			if i < n:
				fvec[i] += x[i]
	elif nprob == 2:  # Linear function - rank 1.
		for j in range(n):
			sum += (j + 1) * x[j]
		for i in range(m):
			fvec[i] = (i+1) * sum - 1
	elif nprob == 3:  # Linear function - ranke 1 with zero columns and rows.
		for j in range(1, n - 1):
			sum += (j + 1) * x[j]
		for i in range(m - 1):
			fvec[i] = i * sum - 1
		fvec[m - 1] = -1
	elif nprob == 4:  # Rosenbrock function.
		fvec[0] = 10 * (x[1] - x[0] ** 2)
		fvec[1] = 1 - x[0]
	elif nprob == 5:  # Helical valley function.
		if x[0] > 0:
			th = arctan(x[1] / x[0]) / (2 * pi)
		elif x[0] < 0:
			th = arctan(x[1] / x[0]) / (2 * pi) + .5
		else:  # x[0] == 0
			th = .25
		r = sqrt(x[0] ** 2 + x[1] ** 2)
		fvec[0] = 10 * (x[2] - 10 * th)
		fvec[1] = 10 * (r - 1)
		fvec[2] = x[2]
	elif nprob == 6:  # Powell singular function.
		fvec[0] = x[0] + 10 * x[1]
		fvec[1] = sqrt(5) * (x[2] - x[3])
		fvec[2] = (x[1] - 2 * x[2]) ** 2
		fvec[3] = sqrt(10) * (x[0] - x[3]) ** 2
	elif nprob == 7:  # Freudenstein and Roth functions.
		fvec[0] = -c13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]
		fvec[1] = -c29 + x[0] + ((1 + x[1]) * x[1] - c14) * x[1]
	elif nprob == 8:  # Bard function.
		for i in range(15):
			tmp1 = i + 1
			tmp2 = 16 - i - 1
			tmp3 = tmp1
			if i > 7:
				tmp3 = tmp2
			fvec[i] = y1[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3))
	elif nprob == 9:  #Kowalik and Osbourne function.
		for i in range(11):
			tmp1 = v[i] * (v[i] + x[1])
			tmp2 = v[i] * (v[i] + x[2]) + x[3]
			fvec[i] = y2[i] - x[0] * tmp1 / tmp2
	elif nprob == 10:  # Meyer function.
		for i in range(16):
			temp = 5 * (i + 1) + c45 + x[2]
			tmp1 = x[1] / temp
			tmp2 = exp(tmp1)
			fvec[i] = x[0] * tmp2 - y3[i]
	elif nprob == 11:  # Watson function.
		for i in range(29):
			div = (i + 1) / c29
			s1 = 0
			dx = 1
			for j in range(1, n):
				s1 += j * dx * x[j]
				dx = div * dx
			s2 = 0
			dx = 1
			for j in range(n):
				s2 = s2 + dx * x[j]
				dx = div * dx
			fvec[i] = s1 - s2 ** 2 - 1
		fvec[29] = x[0]
		fvec[30] = x[1] - x[0] ** 2 - 1
	elif nprob == 12:  # Box 3-dimensional function.
		for i in range(m):
			temp = i + 1
			tmp1 = temp / 10
			fvec[i] = exp(-tmp1 * x[0]) - exp(-tmp1 * x[1]) + (exp(-temp) - exp(-tmp1)) * x[2]
	elif nprob == 13:  # Jennrich and Sampson function.
		for i in range(m):
			temp = i + 1
			fvec[i] = 2 + 2 * temp - exp(temp * x[0]) - exp(temp * x[1])
	elif nprob == 14:  # Brown and Dennis function.
		for i in range(m):
			temp = (i + 1) / 5
			tmp1 = x[0] + temp * x[1] - exp(temp)
			tmp2 = x[2] + sin(temp) * x[3] - cos(temp)
			fvec[i] = tmp1 ** 2 + tmp2**2
	elif nprob == 15:  # Chebyquad function.
		for j in range(n):
			t1 = 1
			t2 = 2 * x[j] - 1
			t = 2 * t2
			for i in range(m):
				fvec[i] = fvec[i] + t2
				th = t * t2 - t1
				t1 = t2
				t2 = th
		iev = -1
		for i in range(m):
			fvec[i] /= n
			if iev > 0:
				fvec[i] += 1 / ((i + 1) ** 2 - 1)
			iev = -iev
	elif nprob == 16:  # Brown almost-linear function.
		sum1 = -(n+1)
		prod1 = 1
		for j in range(n):
			sum1 += x[j]
			prod1 *= x[j]
		for i in range(n - 1):
			fvec[i] = x[i] + sum1
		fvec[n-1] = prod1 - 1
	elif nprob == 17:  # Osborne 1 function.
		for i in range(33):
			temp = 10 * i
			tmp1 = exp(-x[3] * temp)
			tmp2 = exp(-x[4] * temp)
			fvec[i] = y4[i] - (x[0] + x[1] * tmp1 + x[2] * tmp2)
	elif nprob == 18:  # Osborne 2 function.
		for i in range(65):
			temp = i / 10
			tmp1 = exp(-x[4] * temp)
			tmp2 = exp(-x[5] * (temp - x[8]) ** 2)
			tmp3 = exp(-x[6] * (temp - x[9]) ** 2)
			tmp4 = exp(-x[7] * (temp - x[10]) ** 2)
			fvec[i] = y5[i] - (x[0] * tmp1 + x[1] * tmp2 + x[2] * tmp3 + x[3] * tmp4)
	elif nprob == 19:  # Bdqrtic
		# n>=5, m = (n - 4) * 2
		for i in range(n - 4):
			fvec[i] = (-4 * x[i] + 3.0)
			fvec[n - 4 + i] = (x[i]**2 + 2 * x[i + 1] ** 2 + 3 * x[i + 2]**2 + 4 * x[i + 3]**2 + 5 * x[n-1] ** 2)
	elif nprob == 20:  # Cube
		# n=2 m=n
		fvec[0] = (x[0] - 1.0)
		for i in range(1, n):
			fvec[i] = 10 * (x[i] - x[i - 1] ** 3)
	elif nprob == 21: # Mancino
		# n >=2 m=n
		for i in range(n):
			ss = 0
			for j in range(n):
				v2 = sqrt(x[i] ** 2 + (i + 1) / (j + 1))
				ss = ss + v2 * ((sin(log(v2)))**5 + (cos(log(v2)))**5)
			fvec[i] = 1400 * x[i] + (i + 1 - 50)**3 + ss
	elif nprob == 22:  # Heart8ls
		# m=n=8
		fvec[0] = x[0] + x[1] + 0.69
		fvec[1] = x[2] + x[3] + 0.044
		fvec[2] = x[4] * x[0] + x[5] * x[1] - x[6] * x[2] - x[7] * x[3] + 1.57
		fvec[3] = x[6] * x[0] + x[7] * x[1] + x[4] * x[2] + x[5] * x[3] + 1.31
		fvec[4] = x[0] * (x[4]**2 - x[6]**2) - 2.0 * x[2] * x[4] * x[6] + x[1] * (x[5]**2 - x[7]**2) - 2.0 * \
				x[3] * x[5] * x[7] + 2.65
		fvec[5] = x[2] * (x[4]**2 - x[6]**2) + 2.0 * x[0] * x[4] * x[6] + x[3] * (x[5]**2 - x[7]**2) + 2.0 * \
				x[1] * x[5] * x[7] - 2.0
		fvec[6] = x[0] * x[4] * (x[4]**2 - 3.0 * x[6]**2) + x[2] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2) + x[1] \
			* x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2) + x[3] * x[7] * (x[7] ** 2 - 3.0 * x[5] ** 2) + 12.6
		fvec[7] = x[2] * x[4] * (x[4] ** 2 - 3.0 * x[6] ** 2) - x[0] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2) + \
			  x[3] * x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2) - x[1] * x[7] * (x[7] ** 2 - 3.0 * x[5] ** 2) - \
			  9.48
	if context is not None:
		context.fvals.append(fvec)
		context.nfev += 1

	return fvec