function [y] = ineq_constraints_3(x, k)

switch k
	case 0
		y = [ 3 - x(1) ];
	case 1
		y = [  -1   0  ];
	otherwise
		disp('error with derivative of constraints');
end

end
