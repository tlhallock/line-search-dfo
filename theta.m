function [violation] = theta(statement, x)

violation = 0;

if isfield(statement, 'h')
	violation = violation + sum(abs(statement.h(x, 0)));
end

if isfield(statement, 'g')
	violation = violation + sum(max(0, statement.g(x, 0)));
end

end
