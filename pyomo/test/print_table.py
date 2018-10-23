import os
import re

from test.run_params import params


lst = []

def tr(s):
	if s == 'circle':
		return 'spherical'
	if s == 'ellipse':
		return 'ellipsoid'
	if s == 'scaled-ellipse':
		return 'scaled ellipsoid'
	if s == 'circumscribed-ellipse':
		return 'circumscribed ellipsoid'
	if s == 'simplex-search':
		return 'max volume'
	return s

for p in params:
	num_evaluations = -1
	num_iterations = -1
	with open(os.path.join('images', p['directory'], 'log.txt'), 'r') as logfile:
		for line in logfile:
			m = re.search('total number of evaluations = ([0-9]*)', line)
			if m:
				try:
					num_evaluations = max(num_evaluations, int(m.group(1)))
				except:
					continue

			m = re.search('iteration = ([0-9]*)', line)
			if m:
				try:
					num_iterations = max(num_iterations, int(m.group(1)))
				except:
					continue

	lst.append((
		p['trust_region_options']['shape'],
		p['trust_region_options']['search'] + ('*' if 'heuristics' in p['trust_region_options'] else ''),
		p['trust_region_options']['number_of_points'] if 'number_of_points' in p['trust_region_options'] else ' ',
		p['basis'],
		num_iterations,
		num_evaluations
	))


pl = None
for t in sorted(lst):
	l = [
		str(v)
		for v in t
	]

	ml = []
	changed = False
	changed_idx = 10000
	for i in range(len(l)):
		changed = changed or pl is None or pl[i] != l[i]
		if changed:
			ml.append(l[i])
			changed_idx = min(changed_idx, i)
		else:
			ml.append('')
	# \\cline{' + str(changed_idx) + '-6}'

	lens = [25, 10, 5, 10, 5, 5]
	tp = ml
	print(' & '.join([tr(tp[i]).rjust(lens[i]) for i in range(len(tp))]) + ' \\\\')
	pl = l



