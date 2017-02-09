function [y] = positive_definite_quadratic(x, k, center, Q, b)


if nargin < 3
	center = [0; 0];
end

if nargin < 4
	%Q = magic(length(x));
	Q = [3 0; 0 .5];
	Q = Q + Q';
	Q = Q / norm(Q);
end

if nargin < 5
	b = (1:length(x))';
	b = zeros(2,1);
end

x = x - center;

switch k
	case 0
		y = .5 * x' * Q * x + b' * x;
	case 1
		y = Q * x + b;
	case 2
		y = Q;
	otherwise
		disp('Error in objective derivative');

end
