function [yes] = check_stopping_criteria(statement, state)

yes = false;

if abs(norm(state.g)) < statement.tol
	yes = true;
	return;
end


if isfield(statement, 'g') && isfield(statement, 'h')

else if isfield(statement, 'g')
        false && ~isfield(statement, 'h')
else if isfield(statement, 'h')

else
	% unconstrained minimization problem!
	lambda = -state.A' \ state.g;
	if norm(state.g + state.A'*lambda) < statement.tol
		yes = true;
		return;
	end
end


% state.active = state.cIneq > -statement.tol


	
		% feasible
%		all(c < statement.tol) ...
%		&& all(...
%		% gradient goes into infeasible region
%		(abs(A*g) < statement.tol) ...
%		% but only needed for active contraints
%		.* (1-(abs(c) < statement.tol))...
%	if ...
%		all(abs(state.c) < statement.tol) ...
%			&& arecolinear(abs(state.A(i,:)*state.g/(norm(state.A)*norm(state.g))) < statement.tol) ...
%		break;
%	end
	
	
	
	



end
