
import json

from trust_region.optimization.maximize_multidimensional_ellipse import compute_maximal_ellipse
from trust_region.optimization.maximize_multidimensional_ellipse import parse_params
from trust_region.optimization.maximize_multidimensional_ellipse import construct_ellipse

from trust_region.util.bounds import Bounds
from trust_region.util.plots import create_plot

import numpy as np

params = '''
{
  "center": [
    0.0,
    0.0
  ],
  "A": [
    [
      -36.70522143457306,
      0.0
    ],
    [
      0.0,
      -12.83850079367221
    ],
    [
      1.0,
      0.0
    ],
    [
      -1.0,
      0.0
    ],
    [
      0.0,
      1.0
    ],
    [
      0.0,
      -1.0
    ]
  ],
  "b": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "include-point": null,
  "tolerance": 1e-08,
  "hot-start": null
}
'''










parameters = parse_params(json.loads(params))
success, ellipse = compute_maximal_ellipse(parameters)

bounds = Bounds()
bounds.extend(np.array([1, 1]))
bounds.extend(np.array([-1, -1]))
p = create_plot('an ellipse', 'what_is_wrong.png', bounds)
ellipse.add_to_plot(p)

test_matrix = np.array([
	[np.sqrt(1 / 36 ** 2), 0],
	[0, np.sqrt(1 / 11 ** 2)]
]) / 2

# construct_ellipse(
# 	parameters,
# 	test_matrix,
# 	None,
# 	1.0,
# 	None,
# 	None
# ).add_to_plot(p)

p.add_polyhedron(parameters.A, parameters.b, label='the constraints')

p.save()

print(
	json.dumps(
	ellipse.to_json(),
		indent=2
	)
)




from trust_region.util.polyhedron import get_polyhedron
from trust_region.util.polyhedron import shrink

some_points = np.random.random((4, 2))
center = np.mean(some_points, axis=0)
A, b = get_polyhedron(some_points)


bounds = Bounds()
bounds.extend(np.array([1, 1]))
bounds.extend(np.array([0, 0]))
p2 = create_plot('an ellipse', 'shrunk_polyhedron.png', bounds)
p2.add_point(center, label='center', color='g')
p2.add_polyhedron(A, b, label='first', color='b')
As, bs = shrink(A, b, center, 0.9)
p2.add_polyhedron(As, bs, label='second', color='r')
p2.save()