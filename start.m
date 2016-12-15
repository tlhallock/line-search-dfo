

n = 2;

addpath('nondom');
addpath('functions');

statement = struct();
statement.f = @(x, k) (objective_1(x, k));
statement.g = @(x, k) (ineq_constraints_3(x, k));
statement.h = @(x, k) (eq_constraints_2(x, k));
statement.theta = @(x) (                 ...
	norm(statement.h(x, 0))          ...
	+ sum(max(0, statement.g(x, 0))) ...
);
statement.x0 = [.5; .5];
statement.tol = 1e-6;

constants = create_constants(statement);

results = algorithm(statement, constants);

