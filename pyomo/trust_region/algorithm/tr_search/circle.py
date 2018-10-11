import numpy

from trust_region.util.trust_region import CircularTrustRegion
from trust_region.algorithm.tr_search.trust_region_strategy import TrustRegionStrategy


class CircularTrustRegionStrategy(TrustRegionStrategy):
	def find_trust_region(self):
		a, b = self.context.get_polyhedron()
		distance_to_closest_constraint = min(
			# There is something wrong with the two abs here, the inner abs should not be needed
			numpy.divide(abs(numpy.dot(a, self.context.model_center()) - b), numpy.linalg.norm(a, axis=1))
		)
		return CircularTrustRegion(
			center=self.context.model_center(),
			radius=distance_to_closest_constraint
		)

	def add_to_plot(self, plot_object):
		pass
