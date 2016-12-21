function [y] = rosenbrock(x, k, a, b)

if nargin < 3
	a = 1;
end

if nargin < 4
	b = 100;
end

switch k
	case 0
		y = (x(1) - a)^2 + b*(x(2) - x(1)^2)^2;
	case 1
		y = [	2*(x(1) - a) + 2*b*(x(2) - x(1)^2) * (-2*x(1));	...
			           0 + 2*b*(x(2) - x(1)^2) * 1		...
		];
	case 2
		y = [
			2*x(1) - 4*b*(x(2)-2*x(1)) ...
			2*b*(-2*x(1)) ; ...
			-4*b*x(1) ;	...
			2*b		...
			];
	otherwise
		disp('Error in objective derivative');
end
