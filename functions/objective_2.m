function [y] = objective_2(x, k)


switch k
	case 0
		y = x * x;
	case 1
		y = 2 * x;
	case 2
		y = 2;
	otherwise
		disp('Error in objective derivative');
end
