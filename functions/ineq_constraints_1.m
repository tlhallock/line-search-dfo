function [y] = ineq_constraints_1(x, k)

switch k
	case 0
		y = [ (1-x(1))^2 + (2-x(2))^2 - 3 ; ...
			 x(2) - 2 - 3*(x(1) - 1)       ...
		];
	case 1
		y = [ ...
			2*(1-x(1)) * -1, 2*(2-x(2))*-1; ...
			-3, 1 ...
		];

	otherwise
		disp('error with derivative of constraints');
end

end
