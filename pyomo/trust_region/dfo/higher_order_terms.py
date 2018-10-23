
import numpy

from trust_region.optimization.sample_higher_order_mins import minimize_other_polynomial
from trust_region.util.bounds import Bounds
from trust_region.util.plots import create_plot
from trust_region.util.nullspace import nullspace


class ExtraTerms:
	def __init__(self, basis):
		self.basis = basis
		self.null_space = None
		self.phi = None

	def set_model(self, points, phi):
		self.null_space = numpy.asarray(
			nullspace(
				numpy.bmat([
					self.basis.evaluate_to_matrix(points),
					numpy.array([self.evaluate_extra(points[i, :]) for i in range(points.shape[0])])
				])
			)
		)
		self.phi = phi

	def evaluate_extra(self, x):
		return numpy.array([
			(1.0 / 6.0) * x[0] * x[0] * x[0],
			(1.0 / 6.0) * x[1] * x[1] * x[1]
		])

	def debug_evaluate(self, x, sample_c):
		e = numpy.dot(self.null_space, sample_c)
		e_coefficients = e[-self.extra_dimension():]
		return (
			self.basis.debug_evaluate(x, self.phi + e[:-self.extra_dimension()]) +
			(1.0 / 6.0) * x[0] * x[0] * x[0] * e_coefficients[0] +
			(1.0 / 6.0) * x[1] * x[1] * x[1] * e_coefficients[1]
		)

	def to_pyomo_expression(self, model, sample_c):
		e = numpy.dot(self.null_space, sample_c)
		e_coefficients = e[-self.extra_dimension():]
		return (
			self.basis.to_pyomo_expression(model, self.phi + e[:-self.extra_dimension()]) +
			(1.0 / 6.0) * model.x[0] * model.x[0] * model.x[0] * e_coefficients[0] +
			(1.0 / 6.0) * model.x[1] * model.x[1] * model.x[1] * e_coefficients[1]
		)

	def extra_dimension(self):
		return 2


def sample_other_minimums(
		basis,
		objective_ceofficients,
		trust_region,
		points,
		n_samples=10,
		variance=1.0,
		plotting_objective_lambda=None
):
	extra_terms = ExtraTerms(basis)
	extra_terms.set_model(points, objective_ceofficients)

	expected_solution = minimize_other_polynomial(
		basis.n,
		extra_terms,
		trust_region,
		None,
		numpy.zeros(basis.n)
	)

	other_solutions = [expected_solution]
	for sample_no in range(n_samples):
		sampled_coeff = numpy.random.normal(numpy.zeros(basis.n), variance)
		other_solution = minimize_other_polynomial(2, extra_terms, trust_region, expected_solution, sampled_coeff)
		other_solutions.append(other_solution)

		################################################################################################################
		# bounds = Bounds()
		# for p in points:
		# 	bounds.extend(p)
		# plot = create_plot('testing', 'images/other_polynomial_{}.png'.format(sample_no), bounds.expand())
		#
		# plot.add_point(expected_solution, 'expected solution', color='m')
		# plot.add_point(other_solution, 'solution_distributions', color='y')
		# plot.add_points(points, 'poised_points')
		# plot.add_contour(
		# 	lambda x: basis.debug_evaluate(x, objective_ceofficients),
		# 	'objective',
		# 	color='b'
		# )
		# plot.add_contour(
		# 	lambda x: extra_terms.debug_evaluate(x, sampled_coeff),
		# 	'sampled_objective',
		# 	color='r'
		# )
		# plot.add_contour(
		# 	plotting_objective_lambda,
		# 	'sampled_objective',
		# 	color='k'
		# )
		# trust_region.add_to_plot(plot)
		#
		# plot.save()
		################################################################################################################

	####################################################################################################################
	# bounds = Bounds()
	# for p in points:
	# 	bounds.extend(p)
	# plot = create_plot('testing', 'images/sampled_minimums.png', bounds.expand())
	#
	# plot.add_point(expected_solution, 'expected solution', color='m')
	# plot.add_points(points, 'poised_points', color='r')
	# plot.add_points(numpy.array(other_solutions), 'solution_distributions', color='y')
	# plot.add_contour(
	# 	lambda x: basis.debug_evaluate(x, objective_ceofficients),
	# 	'objective',
	# 	color='b'
	# )
	# if plotting_objective_lambda is not None:
	# 	plot.add_contour(
	# 		plotting_objective_lambda,
	# 		'sampled_objective',
	# 		color='k'
	# 	)
	# trust_region.add_to_plot(plot)
	#
	# plot.save()
	####################################################################################################################
