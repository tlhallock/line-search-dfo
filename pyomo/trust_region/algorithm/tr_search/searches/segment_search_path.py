
import itertools
import numpy as np
from scipy.linalg import null_space
from trust_region.util.polyhedron import Polyhedron

import scipy


class SearchPath:
	def __init__(self, points, distances):
		self.points = points
		self.distances = distances

	def __str__(self):
		return 'points: [' + ','.join([str(x) for x in self.points]) + ']\ndistances: [' + ','.join([str(x) for x in self.distances]) + ']'

	def __repr__(self):
		return self.__str__()

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
		plot_object.add_points(np.array(self.points), label='search path', color='r', s=30, marker='o')
		# width = 0.05 * max(np.linalg.norm(self.points[i] - self.points[i + 1]) for i in range(len(self.points) - 1))
		# for i in range(len(self.points) - 1):
		# 	plot_object.add_arrow(
		# 		self.points[i],
		# 		self.points[i + 1],
		# 		width=width
		# 	)


def _get_basis_for_intersection_of_basis(basis1, basis2):
	if basis1.shape[0] == 0 or basis2.shape[0] == 0:
		return np.zeros((0, basis1.shape[1]))
	return null_space(np.hstack([basis1, -basis2]))


def _contains_parallel_faces(As, tolerance):
	for i1, i2 in itertools.combinations([i for i in range(len(As))], 2):
		if np.linalg.norm(As[i1] + As[i2]) < tolerance:
			return True
	return False


def _get_distances(A, b, p):
	return np.divide(abs(A @ p - b), np.linalg.norm(A, axis=1))


def get_search_path(x, polyhedron, num, tolerance):
	print('=================================================')
	print(x)
	print(polyhedron.A)
	print(polyhedron.b)
	print('=================================================')
	An = np.linalg.norm(polyhedron.A, axis=1)
	remaining_b = polyhedron.b / An
	remaining_faces = polyhedron.A / An[:, None]
	all_points = [x]
	equidistant_faces = []
	equidistant_b = []
	while True:
		distances = remaining_faces@all_points[-1] - remaining_b
		min_distance = max(distances)
		equidistant = [
			(remaining_faces[idx], remaining_b[idx])
			for idx, distance in enumerate(distances)
			if abs(distance - min_distance) < tolerance
		]
		remaining = [
			(remaining_faces[idx], distance, remaining_b[idx])
			for idx, distance in enumerate(distances)
			if abs(distance - min_distance) >= tolerance
		]
		equidistant_faces = equidistant_faces + [t[0] for t in equidistant]
		equidistant_b = equidistant_b + [t[1] for t in equidistant]
		remaining_faces = np.array([r[0] for r in remaining])
		remaining_d = np.array([r[1] for r in remaining])
		remaining_b = np.array([r[2] for r in remaining])
		if (
			len(equidistant_faces) == len(x) + 1 or
			len(equidistant_faces) == num + 1 or
			len(remaining_faces) == 0 or
			_contains_parallel_faces(equidistant_faces, tolerance)
		):
			break

		rA = np.array(equidistant_faces)
		rb = np.array(equidistant_b)
		if any([abs(xi) < tolerance for xi in distances]):
			direction = rA.T@np.linalg.solve(rA@rA.T, -np.ones(len(x)))
		else:
			try:
				direction = rA.T@np.linalg.solve(rA@rA.T, rA@all_points[-1]-rb)
			except np.linalg.linalg.LinAlgError:
				raise Exception("Found parallel faces: [" + ";".join([",".join(r) for r in rA@rA.T]) + "]")

		direction /= np.linalg.norm(direction)

		ts = np.divide(min_distance - remaining_d, remaining_faces@direction - equidistant_faces[0]@direction)
		ts = ts[ts > tolerance]
		if len(ts) == 0:
			break
		t = min(ts)

		# print('=============================================================================')
		# print(distances)
		# print(equidistant_faces)
		# print(all_points[-1])
		# print(direction)
		# print(t)
		# print(remaining_faces@direction, equidistant_faces[0]@direction)
		# print(all_points)
		# print(all_points[-1] + t * direction)
		# print('=============================================================================')

		# if np.isinf(t) or np.isnan(t):
		# 	break
		all_points.append(all_points[-1] + t * direction)
	print(all_points)
	return SearchPath(all_points, [_get_distances(polyhedron.A, polyhedron.b, p) for p in all_points])








'''

print(
	get_search_path(
		np.array([2.00000001e+00, 1.03954034e-13]),
		Polyhedron(
			np.array([
				[  1.,   0.],
 				[ -1.,   0.],
 				[  0.,   1.],
 				[  0.,  -1.],
 				[-10.,   1.],
 				[ -1.,   0.],
 				[  0.,  -1.],
 				[  1.,   0.],
 				[  0.,   1.],
			]).astype(float),
			np.array([  2.38698354,  -1.61301649,   0.38698353,   0.38698353, -10,          -2,          50,          50,          50,        ]).astype(float)
		),
		5,
		1e-12
	)
)


		#distances = np.divide(remaining_faces@all_points[-1] - remaining_b,)
	#b = np.divide(b, np.linalg.norm(A, axis=1))
	# for i in range(A.shape[0]):
	# 	n = np.linalg.norm(A[i, :])
	# 	A[i, :] /= n
	# 	b[i] /= n

		#be = np.array(equidistant_b)
		direction = (
			#Ae.T @ np.linalg.solve(Ae@Ae.T, Ae@p-be)
			#if np.linalg.norm(Ae@p-be) > tolerance
			#else
			-np.mean(Ae, axis=0)
		)

		t_min = min(np.divide(min_distance - remaining_d, remaining_faces@direction - equidistant_faces[0]@direction))
		#
		# t_min = None
		# for face, distance, _ in remaining:
		# 	t_intersection = (min_distance - distance) / (face@direction - equidistant_faces[0]@direction)
		# 	if t_intersection > 0 and (t_min is None or t_intersection < t_min):
		# 		t_min = t_intersection

'''