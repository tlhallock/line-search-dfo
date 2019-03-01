import numpy

from trust_region.optimization.sample_higher_order_mins import minimize_other_polynomial
from trust_region.dfo.higher_order_terms import ExtraTerms

from trust_region.util.plots import create_plot


class Heuristics:
	def __init__(self, basis):
		self.basis = basis
		self.previous_coefficients = None
		self.previous_trust_region = None
		self.previous_points = None
		self.previous_trial_point = None
		self.previous_error = None

	def ready_to_sample(self, delta):
		return (
			self.previous_coefficients is not None and
			self.previous_trust_region is not None and
			self.previous_points is not None and
			self.previous_trial_point is not None and
			self.previous_error is not None and
			delta > 0.1
		)

	def update_model_heuristics(self, context, trust_region):
		self.previous_coefficients = context.objective_coefficients
		self.previous_trust_region = trust_region
		self.previous_points = context.sample_points

		self.previous_trial_point = None
		self.previous_error = None

	def update_error_heuristics(self, trial_point, expected_objective, true_objective):
		self.previous_trial_point = trial_point
		self.previous_error = abs(true_objective-expected_objective)

	def sample_objectives(self, context, num_samples):
		current_polyhedron = context.construct_polyhedron()
		extra_terms = ExtraTerms(self.basis)
		variance = 1.0

		points = self.previous_trust_region.shift(self.previous_points)
		extra_terms.set_model(
			points,
			self.previous_coefficients
		)
		shifted_polyhedron = self.previous_trust_region.shift_polyhedron(current_polyhedron)

		expected_solution = minimize_other_polynomial(
			self.basis.n,
			extra_terms,
			shifted_polyhedron,
			None,
			numpy.zeros(self.basis.n)
		)
		other_solutions = [expected_solution]
		for sample_no in range(num_samples):
			sampled_coeff = numpy.random.normal(numpy.zeros(self.basis.n), variance)
			other_solution = minimize_other_polynomial(
				self.basis.n,
				extra_terms,
				shifted_polyhedron,
				expected_solution,
				sampled_coeff
			)
			other_solutions.append(other_solution)
		ret_val = self.previous_trust_region.unshift(numpy.asarray(other_solutions))

		diam = current_polyhedron.get_diameter()

		def value_of_point(x):
			return numpy.mean([
				max(0.01, numpy.exp(-(10.0 * numpy.linalg.norm(x - p) / diam) ** 2))
				for p in ret_val
			])

		def evaluator(indicator):
			N = 1000
			sum = 0
			for point in context.outer_trust_region.sample_unshifted_region(N):
				if not indicator(point):
					continue
				sum += value_of_point(point)
			return sum / N


		################################################################################################################
		bounds = context.outer_trust_region.get_bounds()
		for p in self.previous_points:
			bounds.extend(p)
		for p in ret_val:
			bounds.extend(p)
		plot = create_plot(
			'sampling the previous model',
			'images/{}/{}_sampled_minimums.png'.format(
				context.params.directory,
				str(context.plot_number).zfill(5)
			),
			bounds.expand()
		)
		context.plot_number += 1

		self.previous_trust_region.add_to_plot(plot, detailed=False)
		plot.add_polyhedron(current_polyhedron[0], current_polyhedron[1], 'constraints', color='k')
		plot.add_points(ret_val, 'solution distribution', color='y')
		plot.add_point(self.previous_trust_region.unshift_row(expected_solution), 'expected solution', color='m')
		plot.add_points(self.previous_points, 'poised points', color='r')

		plot.add_contour(
			lambda x: context.basis.debug_evaluate(self.previous_trust_region.shift_row(x), self.previous_coefficients),
			'objective',
			color='b'
		)
		plot.add_contour(
			value_of_point,
			'value',
			color='c'
		)
		if False:
			# WOULD NEED TO SHIFT
			plot.add_contour(
				context.params.objective_function,
				'true_objective',
				color='k'
			)
		plot.save()
		################################################################################################################

		return ret_val, evaluator, value_of_point
