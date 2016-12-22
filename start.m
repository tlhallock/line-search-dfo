

n = 2;

addpath('nondom');
addpath('functions');
addpath('dfo');

global imageNum
imageNum = 0;

statement = struct();
statement.x0 = [-1; -1];
statement.tol = 1e-6;

% Set the objective to be a quadratic
statement.f = @(x, k) (positive_definite_quadratic(x, k));
%statement.f = @(x, k) (rosenbrock(x, k));


if false
	% No constraints
	if isfield(statement, 'h')
		statement = rmfield(statement, 'h');
	end
	if isfield(statement, 'g')
		statement = rmfield(statement, 'g');
	end
	
	statement.name = 'no_constraints';
	results = algorithm(statement, create_constants(statement));
end


% One equality constraint
if false

	if isfield(statement, 'g')
		statement = rmfield(statement, 'g');
	end
	statement.h = @(x, k)(a_line(x, k, [1 ; 1], 3));
	statement.name = 'one_equality';
	results = algorithm(statement, create_constants(statement));
end


% One circular equality constraint
if false
	if isfield(statement, 'g')
		statement = rmfield(statement, 'g');
	end
	statement.h = @(x, k)(distance_to_circle(x, k));
	statement.name = 'one_circular_equality';
	results = algorithm(statement, create_constants(statement));
end


% One inactive inequality constraint
if false
	if isfield(statement, 'h')
		statement = rmfield(statement, 'h');
	end
	statement.g = @(x, k)(a_line(x, k, [1 ; 1], 3));
	statement.name = 'one_inactive_inequality';
	results = algorithm(statement, create_constants(statement));
end


% One active inequality constraint
if false
	if isfield(statement, 'h')
		statement = rmfield(statement, 'h');
	end
	statement.g = @(x, k)(a_line(x, k, [1 ; 1], -3));
	statement.name = 'one_active_inequality';
	results = algorithm(statement, create_constants(statement));
end



% One equality and one active inequality constraint
if false
	statement.h = @(x, k)(a_line(x, k, [1 ; 0], 2));
	statement.g = @(x, k)(circle(x, k, [2 ; 1], 1));
	statement.name = 'equality_active_inequality';
	results = algorithm(statement, create_constants(statement));
end



% One equality and one active inequality and one inactive inequality
if false
	statement.h = @(x, k)(a_line(x, k, [1 ; 0], 2));
	statement.g = @(x, k)([...
		circle(x, k, [2 ; 1], 1) ; ...
		a_line(x, k, [1 ; 1], 3) ]);
	statement.name = 'equality_active_inequality';
	results = algorithm(statement, create_constants(statement));
end



% infeasible
if false
	statement.h = @(x, k)(a_line(x, k, [1 ; 0], 5));
	statement.g = @(x, k)([...
		circle(x, k, [2 ; 1], 1) ; ...
		a_line(x, k, [1 ; 1], 3) ]);
	statement.name = 'equality_active_inequality';
	results = algorithm(statement, create_constants(statement));
end


% non-regular
if false
%	statement.h = ...
%	statement.g = ...
	statement.name = 'equality_active_inequality';
	results = algorithm(statement, create_constants(statement));
end



% One weird equality constraint
% This one has problems right now...
if false
	statement.f = @(x, k) (positive_definite_quadratic(x, k, [3;0]));
	
	
	if isfield(statement, 'g')
		statement = rmfield(statement, 'g');
	end
	statement.h = @(x, k)(sin_constraint(x, k));
	statement.name = 'one_wiggly_equality';
	results = algorithm(statement, create_constants(statement));
end




% One active inequality constraint + one equality constraint
% statement.g = @(x, k) (ineq_constraints_3(x, k));
% statement.h = @(x, k) (eq_constraints_2(x, k));
% statement.name = 'eq_and_active_ineq';
% results = algorithm(statement, constants);


% One equality constraint + one active inequality constraint + one inactive inequality constraint





