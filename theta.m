function [violation] = theta(x)

violation = max(0, sum((5 - x).^2));

end
