import numpy

from trust_region.algorithm.tr_search.trust_region_strategy import TrustRegionStrategy
from trust_region.optimization.maximize_ellipse import compute_maximal_ellipse
from trust_region.optimization.maximize_ellipse import EllipseParams


class EllipticalTrustRegionStrategy(TrustRegionStrategy):
	def find_trust_region(self):
		ellipse_params = EllipseParams()
		ellipse_params.A, ellipse_params.b = self.context.get_polyhedron()
		ellipse_params.center = numpy.copy(self.context.model_center())
		ellipse_params.normalize = False
		ellipse_params.include_point = None
		return compute_maximal_ellipse(ellipse_params)

	def add_to_plot(self, plot_object):
		#plot.add_polyhedron(ellipse_params.A, ellipse_params.b, label='bounds')
		#plot.add_point(ellipse_params.center, label='center', marker='x', color='r')
		#plot.add_point(ellipse_params.include_point, label='center', marker='+', color='y')
		#ellipse.add_to_plot(plot)
		pass
