import numpy
from trust_region.algorithm.tr_search.searches.common import NoPlotDetails


def no_search(context, objective, options):
	x0 = numpy.copy(context.model_center())
	return objective(context, x0, hot_start=None, options=options), NoPlotDetails()
