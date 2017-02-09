function [yes] = check_stopping_criteria(statement, state)

if abs(norm(state.g)) < statement.tol
	yes = true;
	return;
end

if ~isfield(statement, 'g') && ~isfield(statement, 'h')
	% This is the only requirement to check...
	yes = false;
	return;
end




if norm(max(0, state.cIneq)) > statement.tol
	yes = false;
	return;
end

if norm(state.cEq) > statement.tol
	yes = false;
	return;
end

lambda = -state.A' \ state.g;
if norm(state.g + state.A'*lambda) > statement.tol
	yes = false;
	return;
end

% If there are no inequality constraints, then we don't have to check the sign
if ~isfield(statement, 'g')
	yes = true;
	return;
end

% The number of active inequality constraints
nAIneq = sum(state.active);
lambdaIneq = lambda(length(lambda)-nAIneq+1:length(lambda));
% We need this to be positive
if norm(min(0, lambdaIneq)) > statement.tol
	yes = false;
	return;
end

yes = true;
return;


end
