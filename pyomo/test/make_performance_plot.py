
import numpy as np
import matplotlib.pyplot as plt
import itertools


import os
import json
import traceback
import re

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime

np.set_printoptions(linewidth=255)


class Run:
	def __init__(self):
		self.directory = None
		self.algorithm = None
		self.problem = None
		self.dimension = None
		self.status = None
		self.iterations = None
		self.evaluations = None
		self.found_minimum = None
		self.expected_minimum = None
		self.found_minimizer = None
		self.expected_minimizer = None

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return self.to_tex()

	def to_tex(self):
		return " & ".join([
			str.center(self.algorithm, 30),
			str.center(str(self.problem), 5),
			str.center(str(self.dimension), 3),
			str.center(str(self.status), 10),
			str.center(str(self.iterations), 5),
			str.center(str(self.evaluations), 5),
			str.center(str("%.3f" % self.found_minimum), 10),
			str.center(str("%.3f" % self.expected_minimum), 10),
			str.center(str("[" + ",".join(["%.2f" % xi for xi in self.found_minimizer]) + "]"), 5),
			str.center(str("[" + ",".join(["%.2f" % xi for xi in self.expected_minimizer]) + "]"), 5),
		]) + " \\\\"


def create_run(directory, name, result):
	run = Run()
	run.directory = directory
	run.algorithm = name
	run.problem = result['problem']['problem_number']
	run.dimension = result['problem']['n']
	run.status = result['result']['message']
	run.iterations = result['result']['niter']
	run.evaluations = result['result']['neval']
	run.found_minimum = result['result']['minimum']
	run.expected_minimum = result['problem']['minimum']
	run.found_minimizer = result['result']['minimizer']
	run.expected_minimizer = result['problem']['minimizer']

	if name == 'ellipse segment 4' and run.problem == 44:
		return None

	return run


def read_file(base_directory, directory_name, result_file):
	try:
		with open(result_file, 'r') as infile:
			result = json.load(infile)
	except:
		traceback.print_exc()
		return None

	name = str(re.sub('hs_[0-9]*_', '', directory_name).replace('_', ' '))
	if name == 'circumscribed ellipse':
		return None

	return create_run(base_directory, name, result)


def read_directory(base):
	missing = []
	done = []
	for directory_name in sorted(os.listdir(base)):
		directory = os.path.join(base, directory_name)
		if not os.path.isdir(directory):
			continue
		result_file = os.path.join(directory, 'result.json')
		if not os.path.exists(result_file):
			missing.append(directory)
			continue
		r = read_file(base, directory_name, result_file)
		if r is None:
			continue
		done.append(r)

	return done, missing


done1, missing1 = read_directory('/work/research/line-search-dfo/pyomo/current_images')
done2, missing2 = read_directory('/work/research/line-search-dfo/pyomo/images')


done1.sort(key=lambda x: (x.problem, x.algorithm))

print('algorithm', 'problem', 'message', 'n iterations', 'n evaluations', 'found min', 'true min', 'found minimizer', 'true minimizer')
for run in done1:
	print(run)

# for m in missing:
# 	print('kde-open ' + str(m))


# for problem in all_problems:
# 	print(problem)
# 	runs = [r for r in done1 if r.problem == problem]
# 	runs.sort(key=lambda x: x.evaluations)
# 	for r in runs:
# 		print(r)


def get_some_different_colors(n):
	ret = []
	num_nums = int(np.ceil(n ** (1. / 3.))) + 1
	for x1 in np.linspace(0, 1, num_nums):
		for x2 in np.linspace(0, 1, num_nums):
			for x3 in np.linspace(0.05, 0.75, num_nums):
				ret.append([x1, x2, x3])
	return ret


def create_performance_plot(runs):
	all_problems = list(set(x.problem for x in runs))
	all_problems.sort()
	all_algorithms = set(x.algorithm for x in runs)
	some_colors = iter(get_some_different_colors(len(all_algorithms)))

	alg_to_prob_to_evaluations = {
		algorithm: {
			run.problem: run.evaluations
			for run in runs
			if run.algorithm == algorithm
		}
		for algorithm in all_algorithms
	}

	#fig = plt.figure()
	#plt.title('Performance Plot')
	#ax = fig.add_subplot(111)
	#plt.legend(loc='lower left')

	prob_to_idx = np.zeros(max(all_problems)+1)
	for idx in all_problems:
		prob_to_idx[idx] = all_problems.index(idx)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.title('Performance Plot')

	fig.set_size_inches(10, 10)

	min_by_problem = {
		problem: min(run.evaluations for run in runs if run.problem == problem)
		for problem in all_problems
	}

	algorithm_to_offset = {y: x for x, y in zip(np.linspace(-1, 1, len(all_algorithms)) / 7, all_algorithms)}

	for algorithm in alg_to_prob_to_evaluations:
		evals = [
			(problem, alg_to_prob_to_evaluations[algorithm][problem])
			for problem in all_problems
			if problem in alg_to_prob_to_evaluations[algorithm]
		]
		if len(evals) == 0:
			continue
		y = [e[1] / min_by_problem[e[0]] for e in evals]
		x = [prob_to_idx[e[0]] + algorithm_to_offset[algorithm] for e in evals]
		ax.scatter(x, y, color=next(some_colors), label=algorithm)

	plt.legend()
	plt.xticks(range(len(all_problems)), all_problems)
	ax.grid(True)
	fig.savefig('performance_plot.png')

	plt.show()


def compare(first, second):
	all_problems = list(set(list(x.problem for x in first) + list(x.problem for x in second)))
	all_problems.sort()
	all_algorithms = list(set(list(x.algorithm for x in first) + list(x.algorithm for x in second)))
	some_colors = iter(get_some_different_colors(len(all_algorithms)))

	alg_to_prob_to_evaluations_1 = {
		algorithm: {
			run.problem: abs(run.found_minimum - run.expected_minimum)
			for run in first
			if run.algorithm == algorithm
		}
		for algorithm in all_algorithms
	}

	alg_to_prob_to_evaluations_2 = {
		algorithm: {
			run.problem: abs(run.found_minimum - run.expected_minimum)
			for run in second
			if run.algorithm == algorithm
		}
		for algorithm in all_algorithms
	}
	for algorithm in alg_to_prob_to_evaluations_1:
		diff_for_algorithm = []
		for prob in alg_to_prob_to_evaluations_1[algorithm]:
			eval1 = alg_to_prob_to_evaluations_1[algorithm][prob]
			if algorithm not in alg_to_prob_to_evaluations_1 or prob not in alg_to_prob_to_evaluations_2[algorithm]:
				continue

			eval2 = alg_to_prob_to_evaluations_2[algorithm][prob]
			# print(algorithm)
			# print(prob)
			diff_for_algorithm.append(abs(eval1 - eval2))
			# print(eval1)
			# print(eval2)
		print(algorithm)
		print(np.median(diff_for_algorithm))



create_performance_plot(done1)
compare(done1, done2)
