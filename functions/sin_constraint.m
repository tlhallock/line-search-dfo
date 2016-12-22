function [violation] = sin_constraint(x, k)

f = 2;
a = 1;

switch k
	case 0
		violation = [ (x(2) - a*sin(f*x(1)))^2 ];
	case 1
		violation = [-2*(x(2) - a*sin(f*x(1)))*(a*f*cos(f*x(1))), ...
			2*(x(2) - a*sin(f*x(1))) ];
	otherwise
		disp('error in derivative of sin constraint')
end

end
