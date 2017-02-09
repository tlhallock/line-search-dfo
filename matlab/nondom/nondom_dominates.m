function [yes] = nondom_dominates(x1, x2)

# weakly
yes = all(x1 <= x2);

end;