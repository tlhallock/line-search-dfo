
import numpy

from trust_region.algorithm.tr_search.trust_region_strategy import parse_tr_strategy
from trust_region.dfo.lagrange import LagrangeParams
from trust_region.dfo.lagrange import compute_lagrange_polynomials
from trust_region.optimization.trust_region_subproblem import solve_trust_region_subproblem
from trust_region.util.basis import QuadraticBasis
from trust_region.util.history import History
from trust_region.util.plots import create_plot
from trust_region.util.trust_region import L1TrustRegion

OUTPUT_DIRECTORY = 'images'


class AlgorithmParams:
	def __init__(self):
		self.x0 = None
		self.constraints_A = None
		self.constraints_b = None
		self.objective_function = None
		self.trust_region_strategy_params = None
		self.directory = None

		self.criticality_tolerance = 1e-4
		self.subproblem_constraint_tolerance = 1e-8
		self.subproblem_search_tolerance = 1e-3
		self.tolerance = 1e-4
		self.radius_decrease_factor = 0.75
		self.radius_increase_factor = 1.5
		self.plot_bounds = []


class AlgorithmContext:
	def __init__(self, params, log_file):
		self.log_file = log_file
		self.params = params
		self.basis = QuadraticBasis(len(params.x0))
		self.outer_trust_region = L1TrustRegion(center=params.x0, radius=1.0)
		self.iteration = 0
		self.history = History()
		self.plot_number = 0
		self.evaluation_count = 0
		self.current_plot = None
		self.objective_coefficients = None
		self.current_objective_value = self.evaluate_original_objective(params.x0)

		self.sample_points = numpy.array([
			params.x0 for _ in range(self.basis.basis_dimension)
		])
		self.sample_values = numpy.array([
			self.current_objective_value for _ in range(self.basis.basis_dimension)
		])

		for p in params.plot_bounds:
			self.history.bounds.extend(p)

	def log(self, message):
		print(message)
		self.log_file.write(message + '\n')
		self.log_file.flush()

	def evaluate_original_objective(self, x):
		y = self.params.objective_function.evaluate(x)
		self.evaluation_count += 1
		self.history.add_objective_value(x, y)
		return y

	def model_center(self):
		return numpy.copy(self.outer_trust_region.center)

	def decrease_radius(self):
		self.outer_trust_region.multiply_radius(self.params.radius_decrease_factor)

	def increase_radius(self):
		self.outer_trust_region.multiply_radius(self.params.radius_increase_factor)

	def start_current_plot(self):
		title = 'iteration {}'.format(self.iteration)
		file_name = '{}/{}/{}_iteration.png'.format(
			OUTPUT_DIRECTORY,
			self.params.directory,
			str(self.plot_number).zfill(5)
		)
		self.plot_number += 1
		self.current_plot = create_plot(title, file_name, self.history.get_plot_bounds())

	def finish_current_plot(self, iteration_result):
		self.log(iteration_result)
		self.current_plot.ax.text(
			0.1, 0.1,
			iteration_result,
			horizontalalignment='center',
			verticalalignment='center',
			transform=self.current_plot.ax.transAxes
		)
		self.current_plot.save()
		self.current_plot = None

	def plot_history(self):
		title = 'history'.format(self.iteration)
		file_name = '{}/{}/history.png'.format(OUTPUT_DIRECTORY, self.params.directory)
		self.plot_number += 1
		plot = create_plot(title, file_name, self.history.get_plot_bounds())
		self.history.add_to_plot(plot)
		plot.add_polyhedron(self.params.constraints_A, self.params.constraints_b, 'constraints')
		plot.save()

	def plot_accuracy(self, trust_region):
		title = 'accuracy {}'.format(self.iteration)
		file_name = '{}/{}/{}_accuracy.png'.format(
			OUTPUT_DIRECTORY,
			self.params.directory,
			str(self.plot_number).zfill(5)
		)
		self.plot_number += 1
		accuracy_plot = create_plot(title, file_name, self.outer_trust_region.get_bounds().expand())
		self.outer_trust_region.add_to_plot(accuracy_plot)
		accuracy_plot.add_contour(
			lambda x: self.params.objective_function.evaluate(x),
			label='true objective',
			color='g'
		)
		accuracy_plot.add_contour(
			lambda x: self.basis.debug_evaluate(trust_region.shift_row(x), self.objective_coefficients),
			label='modelled objective',
			color='y'
		)
		trust_region.add_to_plot(accuracy_plot, detailed=False)
		accuracy_plot.save()

	def get_polyhedron(self):
		return numpy.array(numpy.bmat([
			[self.params.constraints_A],
			[self.outer_trust_region.get_a()]
		])), numpy.array(numpy.bmat([
			self.params.constraints_b,
			self.outer_trust_region.get_b()
		])).flatten()


def check_criticality(context):
	# TO IMPLEMENT
	return numpy.linalg.norm(context.model_center()) < context.params.criticality_tolerance


def update_inner_trust_region(context):
	find_trust_region = parse_tr_strategy(context.params.trust_region_strategy_params)
	success, trust_region, plot_details = find_trust_region(context)
	if not success:
		raise Exception('Unable to find the trust region')

	params = LagrangeParams()

	certification = compute_lagrange_polynomials(
		context.basis,
		trust_region,
		context.sample_points,
		context,
		params
	)
	if not certification.poised:
		raise Exception('Not poised')

	context.sample_points = certification.unshifted
	original_sample_values = numpy.copy(context.sample_values)
	for idx in range(len(certification.indices)):
		if certification.indices[idx] < 0:
			context.sample_values[idx] = context.evaluate_original_objective(
				certification.unshifted[idx]
			)
		else:
			context.sample_values[idx] = original_sample_values[certification.indices[idx]]
			if abs(
				context.sample_values[idx] - context.params.objective_function.evaluate(certification.unshifted[idx])
			) > 1e-4:
				raise Exception('Error shifting values after pivoting in LU algorithm')

	context.objective_coefficients = numpy.asarray(
		certification.lmbda * numpy.asmatrix(context.sample_values).T
	).flatten()

	plot_details.add_to_plot(context.current_plot)
	trust_region.add_to_plot(context.current_plot)
	certification.add_to_plot(context.current_plot)

	return trust_region


def always_feasible_algorithm(params):
	with open('{}/{}/log.txt'.format(OUTPUT_DIRECTORY, params.directory), 'w') as log_file:
		context = AlgorithmContext(params, log_file)
		while True:
			context.iteration += 1
			context.log('----------------------------------------')
			context.log('iteration = {}'.format(context.iteration))
			context.log('total number of evaluations = {}'.format(context.evaluation_count))
			context.start_current_plot()
			context.current_plot.add_polyhedron(context.params.constraints_A, context.params.constraints_b, label='constraints')
			context.outer_trust_region.add_to_plot(context.current_plot)
			context.current_plot.add_point(context.model_center(), label='center', color='y', marker='o', s=30)

			if check_criticality(context):
				if context.outer_trust_region.radius < context.params.tolerance:
					context.finish_current_plot('converged')
					break
				context.decrease_radius()
				context.finish_current_plot('critical, radius too large')
				continue

			trust_region = update_inner_trust_region(context)
			context.plot_accuracy(trust_region)

			solution = solve_trust_region_subproblem(
				objective_basis=context.basis,
				objective_coefficients=context.objective_coefficients,
				model_center=context.model_center(),
				outer_trust_region=context.outer_trust_region,
				trust_region=trust_region
			)

			context.current_plot.add_arrow(context.model_center(), solution.trial_point, color='m')

			trial_objective_value = context.evaluate_original_objective(solution.trial_point)
			trial_model_value = solution.predicted_objective_value
			current_objective_value = context.current_objective_value
			current_model_value = context.current_objective_value

			rho = (
				current_objective_value - trial_objective_value
			) / (
				current_model_value - trial_model_value
			)
			context.log('new function value = {}'.format(trial_objective_value))
			context.log('trial point = {}'.format(solution.trial_point))
			context.log('rho = {}'.format(rho))

			if rho < 0.1:
				context.decrease_radius()
				context.finish_current_plot('poor model')
				continue

			delta = numpy.linalg.norm(context.model_center() - solution.trial_point)
			if delta < context.outer_trust_region.radius / 8:
				context.decrease_radius()
				context.finish_current_plot('step too small')
				continue

			if rho > 0.9:
				context.increase_radius()

			context.outer_trust_region.recenter(solution.trial_point)
			context.current_objective_value = trial_objective_value

			context.finish_current_plot('accepted')
		context.plot_history()
