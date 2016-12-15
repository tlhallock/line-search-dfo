function [ret] = penalize(statement, x)

ret = 0;
if isfield(statement, 'h')
	violation = statement.h(x, 0);
	ret = ret + sum(violation .* violation);
end

if isfield(statement, 'g')
	violation = statement.g(x, 0);
	M = 1000;
	penalized = zeros(size(violation));
	for i = 1:length(violation)
		if violation(i) > 0
			penalized(i) = violation(i)*violation(i) + violation(i) + 1;
		else if violation(i) < -M
			penalized(i) = (violation + M);
			penalized(i) = penalized(i) * penalized(i);
		else
			penalized(i) = exp(violation(i));
		end
	end
	
	ret = ret + sum(penalized);
end

end
