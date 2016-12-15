function plotstate(statement, results, state, filename)
	a = 5;
	b = 5;
        
	xs = linspace(state.x(1) - a, state.x(1) + a);
	ys = linspace(state.x(2) - b, state.x(2) + b);
	[X,Y] = meshgrid(xs, ys);
	
	
	
	% Plot the objective
	Z = zeros(length(ys), length(xs));
	for i = 1:size(Z, 2)
		for j = 1:size(Z, 1)
			Z(j, i) = statement.f([xs(i); ys(j)], 0);
		end
	end
	
	hf = figure();
	grid on
	contourf(X, Y, Z)
	hold on
	contour(X, Y, Z, [state.f state.f_new])
	
	
	
	% Plot the equality constraints
	if isfield(statement, 'h')
		constr = zeros(length(xs), length(ys), length(state.cEq));
		for i = 1:size(Z, 2)
			for j = 1:size(Z, 1)
				constr(j,i,:) = statement.h([xs(i); ys(j)], 0);
			end
		end
		
		for k = 1:length(state.cEq)
			for i = 1:size(Z, 2)
				for j = 1:size(Z, 1)
					Z(i, j) = constr(i, j, k);
				end
			end
			contour(X, Y, Z, [0, state.cEq(k), 1]);
		end
	end
	
	
	
	% Plot the inequality constraints
	if  isfield(statement, 'g')
		constr = zeros(length(xs), length(ys), length(state.cIneq));
		for i = 1:size(Z, 2)
			for j = 1:size(Z, 1)
				constr(j,i,:) = statement.g([xs(i); ys(j)], 0);
			end
		end
		
		for k = 1:length(state.cIneq)
			for i = 1:size(Z, 2)
				for j = 1:size(Z, 1)
					Z(i, j) = constr(i, j, k);
				end
			end
			contour(X, Y, Z, [0, state.cIneq(k), 1]);
		end
	end
	
	
	
	
	
	
	plot([state.x(1)], [state.x(2)], '*')
	%plot([state.x(1) + state.g(1)], [state.x(2)+state.g(2)], '.')
	q = quiver(state.x(1), state.x(2), -state.g(1), -state.g(2))
	q = quiver(state.x(1), state.x(2), state.xnew(1) - state.x(1), state.xnew(2) - state.x(2))
	
	saveas(hf, strcat(filename, '.png'), 'png');
	
	hold off
	
	close(hf);
	
	
	
	filterset = results.filter.set;
	save(strcat(filename, '.mat'), 'filterset');
end
