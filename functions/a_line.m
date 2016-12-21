function [y] = a_line(x, k, normal, lvl)

if nargin < 3
	normal = [ 1 ; 4 ];
end

if nargin < 4
	lvl    = 5;
end

switch k
	case 0
		y = [ normal' * x - lvl];
	case 1
		y = normal';

	otherwise
		disp('error with derivative of constraints');
end

end
