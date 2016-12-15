function [results] = algorithm(statement, constants)

results = struct();
results.iters = 0;
results.feasibility_iters = 0;
results.x_optimal = 0;
results.f_min = 0;
results.filter = nondom_create();

state = struct();

imageNum = 0;

state.x = statement.x0;
while true
%	fflush(stdout)
	drawnow('update')
	
%	encapsulating in state to make it easier to plot
	state.f = statement.f(state.x, 0);
	state.g = statement.f(state.x, 1);
	state.H = statement.f(state.x, 2);
	
	state.cEq = statement.h(state.x, 0);
	state.AEq = statement.h(state.x, 1);
	
	state.cIneq = statement.g(state.x, 0);
	state.AIneq = statement.g(state.x, 1);
	
	state.active = state.cIneq > -statement.tol;
	
	state.c = [state.cEq ; state.cIneq(state.active)];
	state.A = [state.AEq ; state.AIneq(state.active, :)];
	
	
	n = length(state.x);
	m = length(state.c);
	state.theta = statement.theta(state.x);
	
	
	if check_stopping_criteria(statement, state)
		break;
	end
	
	
	
	
	
	% compute search direction
	% If this is too ill-conditioned, we need to perform feasibility restoration
	kktmat = [state.H state.A' ; state.A zeros(m, m)];
	state.condition = cond(kktmat);
	
	if cond(kktmat) > 10000 && false
		disp('This is too ill-conditioned:');
		disp(cond(kktmat));
		
		state.x = restore_feasibility(statement, state.x);
		results.feasibility_iters = results.feasibility_iters+1;
		continue;
	end
	vec = -kktmat \ [state.g ; state.c];
	state.d = vec(1:n);
	
	
	% backtracking line search
	if state.g'*state.d < -statement.tol
		state.alpha_min = constants.gamma_alpha * min([constants.gamma_theta, ...
			-constants.gamma_f*state.theta/(state.g'*state.d), ...
			constants.delta*(state.theta)^(constants.s_theta)/(-state.g'*state.d)^(constants.s_f)]);
	else
		state.alpha_min = constants.gamma_alpha * constants.gamma_theta;
	end
	
	
	
	state.alpha = 1;
	state.accept = false;
	while ~state.accept
		if state.alpha < state.alpha_min
			state.x = restore_feasibility(statement, state.x);
			results.feasibility_iters = results.feasibility_iters + 1;
			break;
		end
		
		% check filter
		state.xnew = state.x + state.alpha * state.d;
		state.theta_new = statement.theta(state.xnew);
		state.f_new = statement.f(state.xnew, 0);
		
		plotstate(statement, results, state, ...
			strcat('output/image', num2str(imageNum)));
		imageNum = imageNum + 1;
		
		if nondom_isdom(results.filter, [state.theta_new; state.f_new])
			state.alpha = state.alpha * (constants.tau_one + constants.tau_two)/2;
			continue;
		end
		
		%m = @(x) (.5*x'*H*x+g'*x+f);
		m = @(a) (a*state.g'*state.d);
		state.ftype = m(state.alpha) < 0 && ...
			((-m(state.alpha))^constants.s_f * state.alpha^(1-constants.s_f) > constants.delta*state.theta^(constants.s_theta));
		if state.ftype
			if state.f_new <= state.f + constants.eta_f * m(state.alpha)
				state.accept = true;
				break;
			end
		else
			eighta = state.theta_new <= (1-constants.gamma_theta)*state.theta;
			eightb = state.f_new <= state.f - constants.gamma_f*state.theta_new;
			if eighta || eightb
				state.accept = true;
				break;
			end
		end
		
		state.alpha = state.alpha * (constants.tau_one + constants.tau_two)/2;
	end
	
	
	if state.accept
		if state.ftype && (1-constants.gamma_theta)*state.theta_new > statement.tol
			results.filter = nondom_add(results.filter, [(1-constants.gamma_theta)*state.theta_new; state.f-constants.gamma_f*state.theta_new]);
		end
		
		state.x = state.xnew;
		results.f_min = state.f;
		results.x_optimal = state.x;
	end
	
	state
	restults.iters = results.iters + 1;
end





end

