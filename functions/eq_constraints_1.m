function [y] = eq_constraints_1(x, k)

switch k
	case 0
		y = [x(1) - x(2) + 1];
	case 1
		y = [1  -1];

	otherwise
		disp('error with derivative of constraints');
end

end
