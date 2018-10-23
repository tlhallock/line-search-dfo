import numpy

from trust_region.util.enumerate_polyhedron import enumerate_vertices_of_polyhedron
from trust_region.algorithm.tr_search.searches.common import ObjectiveValue
from trust_region.optimization.minimize_circumscribed_ellipse import minimize_ellipse
from trust_region.dfo.trust_region.scaled_ellipse import ScaledEllipse


def get_circumscribed_ellipse_trust_region_objective(context, x, hot_start, options):
	A, b = context.get_polyhedron()
	vertices = numpy.array([v for v, _ in enumerate_vertices_of_polyhedron(A, b)])
	ellipse = minimize_ellipse(vertices, 1e-4)
	value = ObjectiveValue()
	value.point = x
	value.trust_region = ScaledEllipse(ellipse, 1.0, A, b)
	value.success = True
	value.objective = ellipse.volume
	return value
