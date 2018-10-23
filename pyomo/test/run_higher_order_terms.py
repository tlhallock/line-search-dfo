
import numpy
import json

from trust_region.dfo.lagrange import compute_lagrange_polynomials
from trust_region.dfo.trust_region.circular_trust_region import CircularTrustRegion
from trust_region.optimization.sample_higher_order_mins import minimize_other_polynomial
from trust_region.util.bounds import Bounds
from trust_region.util.basis import QuadraticBasis
from trust_region.util.plots import create_plot
from trust_region.util.nullspace import nullspace
from trust_region.dfo.higher_order_terms import ExtraTerms
from trust_region.dfo.higher_order_terms import sample_other_minimums

numpy.set_printoptions(linewidth=255)


def objective(x):
	# numpy.cos(x[0] * x[0] - 2 * x[1] * x[1])
	return x[0] - x[1] + (x[0] ** 2 + 2 * x[1] ** 2) - 0.5 * x[0] ** 3


dim = 2
basis = QuadraticBasis(dim)
trust_region = CircularTrustRegion(numpy.array([0.0, 0.0]), 1)
certification = compute_lagrange_polynomials(
	basis,
	trust_region,
	numpy.zeros((basis.basis_dimension, dim)),
	{'strategy': 'fixed-xsi'}
)
sample_points = certification.unshifted
sample_other_minimums(
	basis,
	numpy.asarray(
		certification.lmbda * numpy.asmatrix(numpy.array([objective(p) for p in sample_points])).T
	).flatten(),
	trust_region,
	sample_points,
	plotting_objective_lambda=objective
)









