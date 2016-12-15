function [y] = eq_constraints_2(x, k)

switch k
	case 0
		y = [...
			((x(1)-3)^2 + (x(2)-3)^2 - 9)^2
		];
	case 1
		y = [...
		2*((x(1)-3)^2 + (x(2)-3)^2 - 9)*2*(x(1)-3) ...
		2*((x(1)-3)^2 + (x(2)-3)^2 - 9)*2*(x(2)-3) ...
		];

	otherwise
		disp('error with derivative of constraints');
end

end
