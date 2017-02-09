#
#
# from numpy import inf as infinity
#
#
#
#
# class Params:
# 	def __init__(self):
# 		self.eta0 = 1/10;
# 		self.eta1 = .5;
# 		self.gamma = .5;
# 		self.gamma_inc = 1.5;
#
# 		self.eps_c = 5e-2;
# 		self.lambda_max = 10;
#
# 		self.mu = 5;
# 		self.beta = 1;
# 		self.omega = .5;
# 		self.radius_max = 50;
#
# 		self.max_iters = 100;
#
# 		self.tolerance = 1e-12;
#
# 		self.xsi = 1e-4;
#
# 		self.outer_trust_region = 2;
# 		self.max_improvement_iterations = 100;
#
#
# class Results:
# 	def __init__(self):
# 		self.fvals = 0
# 		self.xmin = None
# 		self.improvements = 0
# 		self.iterations = 0
# 		self.fmin = infinity
#
#
# def ten_four(program, params):
#
# 	results = Results()
# 	while True:
#
# [s, fail] = algo_update_model(params, s);
# if fail
# 	% If
# 	not poised, improve
# 	the
# 	model
# 	s = algo_model_improve(params, s);
# end
#
# % test_interpolation(params, s);
# while norm(s.g) < params.eps_c...
# 	& & (~s.fullyLinear | | s.radius > params.mu * norm(s.g))
# % Could
# have
# something
# about
# s.radius < params.tolerance?
# s.radius = min(...
# max(s.radius * params.omega, params.beta * norm(s.g)), ...
# s.radius);
# s = algo_model_improve(params, s);
# s.plot_number = plot_state(s, params);
#
# s.radius
#
# results = algo_update_results(s, results);
#
# if s.radius < params.tolerance
# 	return
# end
# end
# extrema = params.interp_extrema(s.model_coeff);
#
# currentX = s.model_center;
# currentVal = mock_structure_get(s.vals, currentX
# ');
#
# newX = unshift_set(extrema.minX
# ', s.model_center', s.radius)';
# [s.vals, newVal] = mock_structure_add(s.vals, newX
# ', s.f, s.radius);
# newVal = newVal
# ';
#
# results.xs = [results.xs;
# newX
# '];
#
# rho = (currentVal - newVal) / ... \
# 	(s.model((currentX - s.model_center) / s.radius) - extrema.minVal);
#
# rho
#
# s.plot_number = plot_state(s, params, newX);
#
# if rho >= params.eta1
# s.radius = min(params.gamma_inc * s.radius, params.radius_max);
# s.model_center = newX;
# s.fullyLinear = false;
# else
# if s.fullyLinear
# 		% I don't like this:
# % y not check take any improvement better than eta1, we are going to improve the model anyway...
# % (Move this if statement outside the fully linear check.)
# if rho >= params.eta0
# s.model_center = newX;
# end
# s.radius = params.gamma * s.radius;
# s.fullyLinear = false;
# else
# s = algo_model_improve(params, s);
# end
# end
#
# % s = algo_clean_poised_set(s);
# results = algo_update_results(s, results);
# end
#
#
# 	return results
#
#
