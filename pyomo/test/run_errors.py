
import os
import json
import numpy

from trust_region.optimization.maximize_2d_ellipse import compute_maximal_ellipse
from trust_region.optimization.maximize_2d_ellipse import parse_params
import trust_region.optimization.maximize_2d_ellipse
from trust_region.util.history import Bounds
from trust_region.util.plots import create_plot

trust_region.optimization.maximize_2d_ellipse.LOG_ERRORS = False

base = 'images/errors/'

for file in os.listdir(base):
	if file.find('.json') < 0:
		continue
	filename = os.path.join(base, file)
	print(filename)
	try:
		with open(filename) as input_file:
			js = json.load(input_file)
	except:
		os.remove(filename)
		continue

	p = parse_params(js)

	bounds = Bounds()
	b = 1e-1
	bounds.extend(numpy.array([+b, +b]))
	bounds.extend(numpy.array([-b, -b]))

	plot = create_plot('ellipse_error', os.path.join(base, file + '.png'), bounds)
	plot.add_polyhedron(p.A, p.b, label='bounds')
	plot.add_point(p.center, label='center', marker='x', color='r')
	if p.include_point is not None:
		plot.add_point(p.include_point, label='include', marker='+', color='y')
	#ellipse.add_to_plot(plot)
	plot.save()

	print(json.dumps(js, indent=2))
	compute_maximal_ellipse(p)


