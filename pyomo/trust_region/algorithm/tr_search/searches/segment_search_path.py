
import numpy
import operator


class SearchPath:
	def __init__(self, points):
		self.points = points

	def get_point(self, t):
		if len(self.points) == 1:
			return self.points[0]

		idx = min(
			int(t * (len(self.points) - 1)),
			len(self.points) - 2
		)
		l1 = (idx + 0) / (len(self.points) - 1)
		l2 = (idx + 1) / (len(self.points) - 1)
		between = (l2 - l1)

		return (l2 - t) / between * self.points[idx] + (t - l1) / between * self.points[idx + 1]

	def add_to_plot(self, plot_object):
		for i in range(len(self.points) - 1):
			plot_object.add_arrow(
				self.points[i],
				self.points[i + 1]
			)


def get_search_path(x, A, b, num):
	# TODO: I NEED TO CLEAN THIS UP
	all_points = [x]

	A = numpy.copy(A).astype(float)
	b = numpy.copy(b).astype(float)
	for i in range(A.shape[0]):
		n = numpy.linalg.norm(A[i, :])
		A[i, :] /= n
		b[i] /= n

	distances = numpy.divide(abs(numpy.dot(A, x) - b), numpy.linalg.norm(A, axis=1))
	s_distances = sorted(enumerate(distances), key=operator.itemgetter(1))
	idx_min = s_distances[0][0]
	idx_sec = s_distances[1][0]

	if abs(s_distances[0][1] - s_distances[1][1]) < 1e-12:
		return SearchPath(all_points)

	Ac = A[idx_min, :]
	Ab = A[idx_sec, :]
	first_direction = -Ac
	second_direction = -(Ac + Ab) / 2
	if numpy.linalg.norm(second_direction) < 1e-12:
		# BANG HEAD AGAINST TABLE
		Ab = A[s_distances[2][0]]
		second_direction = -(Ac + Ab) / 2
		# SOMEHOW NEED TO MAKE SURE THEY ARE LINEARLY INDEPENDENT

	dd = -numpy.dot(A, first_direction)

	if num == 0:
		t_min = None
		for i in range(A.shape[0]):
			if abs(dd[i]) < 1e-12:
				continue

			t_intersection = distances[i] / dd[i]
			if abs(t_intersection) < 1e-12:
				t_min = 0
			if t_intersection >= 0 and (t_min is None or t_intersection < t_min):
				t_min = t_intersection

		all_points.append(x + t_min * first_direction)
		return SearchPath(all_points)



	t_min = None
	for i in range(A.shape[0]):
		if abs(dd[idx_min] - dd[i]) < 1e-12:
			continue

		t_intersection = (distances[i] - distances[idx_min]) / (dd[idx_min] - dd[i])
		if abs(t_intersection) < 1e-12:
			t_min = 0
		if t_intersection >= 0 and (t_min is None or t_intersection < t_min):
			t_min = t_intersection

	p1 = x + t_min * first_direction
	all_points.append(p1)

	d2 = numpy.divide(abs(numpy.dot(A, p1) - b), numpy.linalg.norm(A, axis=1))
	#print("After first point", d2)

	dd = -numpy.dot(A, second_direction)
	t_min = None
	for i in range(A.shape[0]):
		if abs(dd[idx_min] - dd[i]) < 1e-12:
			continue
		t_intersection = (d2[i] - d2[idx_min]) / (dd[idx_min] - dd[i])
		if abs(t_intersection) < 1e-12:
			t_min = 0
		if t_intersection >= 0 and (t_min is None or t_intersection < t_min):
			t_min = t_intersection

	p2 = p1 + t_min * second_direction
	if num == 2:
		all_points.append(p2)

	d3 = numpy.divide(abs(numpy.dot(A, p2) - b), numpy.linalg.norm(A, axis=1))
	#print("After second point", d3)

	return SearchPath(all_points)
