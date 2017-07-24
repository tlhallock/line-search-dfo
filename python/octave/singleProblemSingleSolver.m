function H = singleProblemSingleSolver(fvals)

H = zeros(length(fvals), 1, 1);
H(:, 1, 1) = fvals;

end