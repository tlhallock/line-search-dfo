function [y] = circle(x, k, center, r)

if nargin < 3
	center = [1 ; 2];
end

if nargin < 4
	r  = sqrt(3);
end

switch k
	case 0
		y = [ (center(1)-x(1))^2 + (center(2)-x(2))^2 - r^2];
	case 1
		y = [ -2*(center(1)-x(1)), -2*(center(2)-x(2))];

	otherwise
		disp('error with derivative of constraints');
end

end
