
import numpy
import json

from trust_region.algorithm.tr_search.trust_region_strategy import parse_tr_strategy
from trust_region.dfo.lagrange import compute_lagrange_polynomials
from trust_region.optimization.trust_region_subproblem import solve_trust_region_subproblem
from trust_region.optimization.criticallity import compute_projection
from trust_region.util.basis import parse_basis
from trust_region.util.history import History
from trust_region.util.plots import create_plot
from trust_region.dfo.trust_region.l1_trust_region import L1TrustRegion
from trust_region.algorithm.tr_search.shape.circle import get_circular_trust_region_objective
from trust_region.algorithm.tr_search.searches.common import NoPlotDetails
from trust_region.algorithm.tr_search.heuristics import Heuristics
from trust_region.util.utils import write_json

OUTPUT_DIRECTORY = 'images'


class AlgorithmParams:
	def __init__(self):
		self.x0 = None
		self.constraints_polyhedron = None
		self.objective_function = None
		self.trust_region_strategy_params = None
		self.point_replacement_params = None
		self.directory = None
		self.basis_type = None
		self.buffer_factor = None

		self.criticality_tolerance = 1e-8
		self.subproblem_constraint_tolerance = 1e-8
		self.subproblem_search_tolerance = 1e-6
		self.tolerance = 1e-7
		self.radius_decrease_factor = 0.25
		self.radius_increase_factor = 1.5
		self.rho_upper = 0.9
		self.rho_lower = 0.1
		self.plot_bounds = []


class AlgorithmContext:
	def __init__(self, params):
		self.params = params
		self.basis = parse_basis(params.basis_type, len(params.x0))
		self.outer_trust_region = L1TrustRegion(center=params.x0, radius=1.0)
		self.iteration = 0
		self.history = History()
		self.plot_number = 0
		self.evaluation_count = 0
		self.current_plot = None
		self.objective_coefficients = None
		self.current_objective_value = self.evaluate_original_objective(params.x0)
		self.heuristics = Heuristics(self.basis)

		self.sample_points = numpy.array([
			params.x0 for _ in range(self.basis.basis_dimension)
		])
		self.sample_values = numpy.array([
			self.current_objective_value for _ in range(self.basis.basis_dimension)
		])

		for p in params.plot_bounds:
			self.history.bounds.extend(p)

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
		self.current_plot = create_plot(
			title,
			file_name,
			(
				self.history.get_plot_bounds()
				if len(self.params.plot_bounds) > 0
				else self.outer_trust_region.get_bounds().expand(1.2)
			)
		)

	def finish_current_plot(self, iteration_result):
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
		title = 'history'
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

	def construct_polyhedron(self):
		return (
			self.outer_trust_region.get_polyhedron()
			.intersect(self.params.constraints_polyhedron)
		)

	def construct_buffer(self):
		if self.params.buffer_factor is None:
			return None
		return self.params.constraints_polyhedron.shrink(
			self.model_center(), self.params.buffer_factor
		).shift(
			self.outer_trust_region.center, self.outer_trust_region.radius
		)

	def success(self):
		min_idx = numpy.argmin(self.sample_values)
		return {
			'success': True,
			'message': 'converged',
			'minimum': self.sample_values[min_idx],
			'minimizer': self.sample_points[min_idx, :]
		}

	def create_log_object(self):
		print('Iteration', self.iteration)
		print('Number of evaluations', self.evaluation_count)
		print('Center', self.model_center())
		print('Radius', self.outer_trust_region.radius)
		print('Lowest sample point', min(self.sample_values))
		print('Constraints at center', self.params.constraints_polyhedron.evaluate(self.model_center()))
		print('=============================================')

		return {
			"iteration": self.iteration,
			"number-of-evaluations": self.evaluation_count,
			"previous-sample-points": self.sample_points,
			"previous-sample-values": self.sample_values,
			"outer-trust-region": self.outer_trust_region.to_json(),
			"center": self.model_center()
		}


def check_criticality(context, trust_region, iteration_object):
	x = context.model_center()
	g = context.basis.evaluate_gradient(trust_region.shift_row(x), context.objective_coefficients)
	iteration_object['gradient'] = g
	polyhedron = context.construct_polyhedron()
	success, proj_x = compute_projection(x - g, polyhedron, context.outer_trust_region)
	if not success:
		return False
	iteration_object['projected-gradient'] = proj_x

	t = numpy.linalg.norm(proj_x - context.model_center())
	critical = t < context.params.criticality_tolerance

	iteration_object['xsi'] = t
	iteration_object['critical'] = critical

	title = '{} criticality'.format(context.iteration)
	file_name = '{}/{}/{}_criticality.png'.format(OUTPUT_DIRECTORY, context.params.directory, str(context.plot_number).zfill(5))
	context.plot_number += 1

	bounds = context.outer_trust_region.get_bounds()
	bounds.extend(x-g)
	plot = create_plot(title, file_name, bounds.expand())
	plot.add_polyhedron(polyhedron, label='feasible region', color='m', lvls=[0.0])
	plot.add_point(context.model_center(), label="c", color='k')
	plot.add_point(x-g, label="x-g", color='y')
	plot.add_point(proj_x, label="x-g", color='r')

	plot.add_arrow(context.model_center(), x-g, color="green", width=0.05 * context.outer_trust_region.radius)
	plot.add_arrow(x-g, proj_x, color="blue", width=0.05 * context.outer_trust_region.radius)
	plot.add_arrow(proj_x, context.model_center(), color="red", width=0.05 * context.outer_trust_region.radius)

	plot.ax.text(
		0.1, 0.1,
		str(critical) + ", " + str(t) + " < " + str(context.params.criticality_tolerance),
		horizontalalignment='center',
		verticalalignment='center',
		transform=plot.ax.transAxes
	)
	plot.save()

	return critical


def update_inner_trust_region(context, log_object):
	if context.outer_trust_region.contained_in(context.params.constraints_polyhedron):
		log_object['tr-contained-in-constraints'] = True
		value = get_circular_trust_region_objective(context, context.model_center(), None, None)
		trust_region = value.trust_region
		success = value.success
		plot_details = NoPlotDetails()
	else:
		log_object['tr-contained-in-constraints'] = False
		find_trust_region = parse_tr_strategy(context.params.trust_region_strategy_params)
		success, trust_region, plot_details = find_trust_region(context)

	if trust_region is None:
		raise Exception('Unable to find trust region')

	log_object["inner-trust-region"] = trust_region.to_json()

	if not success:
		raise Exception('Unable to find the trust region')

	certification = compute_lagrange_polynomials(
		context.basis,
		trust_region,
		context.sample_points,
		context.params.point_replacement_params,
		log_object
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

	log_object["new-sample-points"] = context.sample_points
	log_object["new-sample-values"] = context.sample_values
	log_object['coefficients'] = context.objective_coefficients

	plot_details.add_to_plot(context.current_plot)
	trust_region.add_to_plot(context.current_plot)
	certification.add_to_plot(context.current_plot)

	context.heuristics.update_model_heuristics(context, trust_region)

	return trust_region


def write_log_object(log_file, log_object, needs_comma):
	if needs_comma:
		log_file.write(',')
	log_file.write('\n')
	write_json(log_object, log_file, False)
	return True


def always_feasible_algorithm(params):
	with open('{}/{}/log.json'.format(OUTPUT_DIRECTORY, params.directory), 'w') as log_file:
		log_file.write('{\n\t"iterations": [')
		needs_comma = False
		context = AlgorithmContext(params)
		while True:
			context.iteration += 1

			log_object = context.create_log_object()
			context.start_current_plot()
			context.current_plot.add_polyhedron(context.construct_polyhedron(), label='constraints')
			context.outer_trust_region.add_to_plot(context.current_plot)
			context.current_plot.add_point(context.model_center(), label='center', color='y', marker='o', s=30)

			trust_region = update_inner_trust_region(context, log_object)

			context.plot_accuracy(trust_region)

			if check_criticality(context, trust_region, log_object):
				if context.outer_trust_region.radius < context.params.tolerance:
					log_object['converged'] = True
					context.finish_current_plot('converged')
					break

				log_object['converged'] = False
				log_object['iteration-result'] = 'critical, radius too large'
				needs_comma = write_log_object(log_file, log_object, needs_comma)

				context.decrease_radius()
				context.finish_current_plot(log_object['iteration-result'])
				continue

			solution = solve_trust_region_subproblem(
				objective_basis=context.basis,
				objective_coefficients=context.objective_coefficients,
				model_center=context.model_center(),
				trust_region=trust_region,
				buffer=context.construct_buffer(),
			)

			context.current_plot.add_arrow(context.model_center(), solution.trial_point, color='m', width=0.05 * context.outer_trust_region.radius)

			trial_objective_value = context.evaluate_original_objective(solution.trial_point)
			trial_model_value = solution.predicted_objective_value
			current_objective_value = context.current_objective_value
			current_model_value = context.current_objective_value

			rho = (
				current_objective_value - trial_objective_value
			) / (
				current_model_value - trial_model_value
			)

			log_object['trial-point'] = [xi for xi in solution.trial_point]
			log_object['new-function-value'] = trial_objective_value
			log_object['rho'] = rho

			context.heuristics.update_error_heuristics(solution.trial_point, trial_model_value, trial_objective_value)

			if rho < context.params.rho_lower:
				context.decrease_radius()

				log_object['iteration-result'] = 'poor model'
				needs_comma = write_log_object(log_file, log_object, needs_comma)

				context.finish_current_plot(log_object['iteration-result'])
				continue

			delta = numpy.linalg.norm(context.model_center() - solution.trial_point)
			if delta < context.outer_trust_region.radius / 8:
				context.decrease_radius()

				log_object['iteration-result'] = 'step too small'
				needs_comma = write_log_object(log_file, log_object, needs_comma)

				context.finish_current_plot(log_object['iteration-result'])
				continue

			if rho > context.params.rho_lower:
				context.increase_radius()

			context.outer_trust_region.recenter(solution.trial_point)
			context.current_objective_value = trial_objective_value

			log_object['iteration-result'] = 'accepted'
			needs_comma = write_log_object(log_file, log_object, needs_comma)

			context.finish_current_plot(log_object['iteration-result'])

		context.plot_history()

		log_file.write('\n\t]\n}\n')
		return context.success()
