function [y] = distance_to_circle(x, k, center, r)

if nargin < 3
	center = [3 ; 3];
end


if nargin < 4
	r = 2;
end



switch k
	case 0
		y = [ sum((x - center).^2 - r^2)^2 ];
	case 1
		y = [ ...
		2*((x(1)-center(1))^2 + (x(2)-center(2))^2 - r^2)*2*(x(1)-center(1)), ...
		2*((x(1)-center(1))^2 + (x(2)-center(2))^2 - r^2)*2*(x(2)-center(2)) ...
		];

	otherwise
		disp('error with derivative of constraints');
end

end
