function [newx] = restore_feasibility(statement, startX)

%[newx, fval] = fminunc(@(x)(penalize(statement.g(x, 0))), startX);
%newx = fminunc(@(x)(penalize(statement.h(x, 0))), startX);
newx = fminsearch(@(x)(penalize(statement, x)), startX);

end
