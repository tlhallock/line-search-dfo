
import numpy

from trust_region.dfo.lagrange import LagrangeParams
from trust_region.dfo.lagrange import computeLagrangePolynomials
from trust_region.optimization.trust_region_subproblem import solve_trust_region_subproblem
from trust_region.util.trust_region import CircularTrustRegion
from trust_region.util.trust_region import L1TrustRegion
from trust_region.util.basis import QuadraticBasis
import matplotlib.pyplot as plt

a = 1.0


class AlgorithmParams:
	def __init__(self):
		self.x0 = None
		self.constraints_A = None
		self.constraints_b = None
		self.objective_function = None

		self.criticality_tolerance = 1e-4
		self.tolerance = 1e-4
		self.radius_decrease_factor = 0.5


class AlgorithmContext:
	def __init__(self, params):
		self.params = params
		self.basis = QuadraticBasis(len(params.x0))
		self.outer_trust_region = L1TrustRegion(
			center=params.x0,
			radius=1.0
		)
		self.iteration = 0
		self.sample_points = numpy.array([
			params.x0 for _ in range(self.basis.basis_dimension)
		])
		self.sample_values = numpy.array([
			params.objective_function.evaluate(params.x0) for _ in range(self.basis.basis_dimension)
		])
		self.current_objective_value = params.objective_function.evaluate(params.x0)
		self.history = None
		self.objective_coefficients = None
		self.plot_number = 0

	def model_center(self):
		return self.outer_trust_region.center

	def decrease_radius(self):
		self.outer_trust_region.multiply_radius(self.params.radius_decrease_factor)


def check_criticality(context):
	# TO IMPLEMENT
	return numpy.linalg.norm(context.model_center()) < context.params.criticality_tolerance


def update_inner_trust_region(context):
	distance_to_closest_constraint = min(
		numpy.divide(
			abs(numpy.dot(context.params.constraints_A, context.model_center()) - context.params.constraints_b),
			numpy.linalg.norm(context.params.constraints_A, axis=1)
		)
	)
	trust_region = CircularTrustRegion(
		center=context.model_center(),
		radius=distance_to_closest_constraint
	)

	params = LagrangeParams()

	certification = computeLagrangePolynomials(
		context.basis,
		trust_region,
		context.sample_points,
		context,
		params
	)
	if not certification.poised:
		raise Exception('Not poised')

	context.sample_points = certification.unshifted

	# Could be much smarter
	context.sample_values = numpy.array([
		context.params.objective_function.evaluate(
			certification.unshifted[i]
		) for i in range(certification.unshifted.shape[0])
	])

	context.objective_coefficients = numpy.asarray(
		certification.lmbda * numpy.asmatrix(context.sample_values).T
	).flatten()

	fig = plt.figure()
	fig.set_size_inches(15, 15)
	ax = fig.add_subplot(111)
	trust_region.add_to_plot(ax)
	certification.add_to_plot(ax)
	plt.legend(loc='lower left')
	file_name = 'images/{}_geometry.png'.format(context.plot_number)
	print('saving to {}'.format(file_name))
	fig.savefig(file_name)
	context.plot_number += 1
	plt.close()

	return trust_region


def always_feasible_algorithm(params):
	context = AlgorithmContext(params)
	while True:
		context.iteration += 1

		if check_criticality(context):
			if context.outer_trust_region.radius < context.params.tolerance:
				break
			context.decrease_radius()
			continue

		trust_region = update_inner_trust_region(context)

		solution = solve_trust_region_subproblem(
			objective_basis=context.basis,
			objective_coefficients=context.objective_coefficients,
			model_center=context.model_center(),
			outer_trust_region=context.outer_trust_region,
			trust_region=trust_region
		)

		trial_objective_value = context.params.objective_function.evaluate(solution.trial_point)
		trial_model_value = solution.predicted_objective_value
		current_objective_value = context.current_objective_value
		current_model_value = context.current_objective_value

		rho = (
			current_objective_value - trial_objective_value
		) / (
			current_model_value - trial_model_value
		)
		print('new function value = {}'.format(trial_objective_value))
		print('trial point = {}'.format(solution.trial_point))
		print('rho = {}'.format(rho))
		if rho < .5:
			context.decrease_radius()
			continue

		delta = numpy.linalg.norm(context.model_center() - solution.trial_point)
		if delta < context.outer_trust_region.radius / 8:
			context.decrease_radius()
			continue

		context.outer_trust_region.recenter(solution.trial_point)
		context.current_objective_value = trial_objective_value
