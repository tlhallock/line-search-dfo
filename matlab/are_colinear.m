function [yes] = are_colinear(A, g, tol)

costheta = A*g/(norm(A)*norm(g));

yes = abs(1-abs(costheta)) < tol;

end
