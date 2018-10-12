import numpy


def sample_search_directions(dim, number):
	for i in range(dim):
		ret_val = numpy.zeros(dim)
		ret_val[i] = 1.0
		yield ret_val

		ret_val = numpy.zeros(dim)
		ret_val[i] = -1.0
		yield ret_val

	for i in range(number):
		ret_val = 2 * numpy.random.random(dim) - 1
		ret_val /= numpy.linalg.norm(ret_val)
		yield ret_val
