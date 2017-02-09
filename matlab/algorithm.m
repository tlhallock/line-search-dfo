function [results] = algorithm(statement, constants)

results = create_results();

state = struct();

state.x = statement.x0;

iteration_number = 0;

while true
	% octave:
%	fflush(stdout);
	% matlab:
%	drawnow('update');
	
	
	
	% Calculate all constraint/objective derivativative/value information at this iterate...
	state.f = statement.f(state.x, 0);
	state.g = statement.f(state.x, 1);
	state.H = statement.f(state.x, 2);
	
	if isfield(statement, 'h')
		state.cEq = statement.h(state.x, 0);
		state.AEq = statement.h(state.x, 1);
	else
		% Should set the dimensions of these
		state.cEq = [];
		state.AEq = [];
	end
	
	state.c = state.cEq;
	state.A = state.AEq;
	
	if isfield(statement, 'g')
		state.cIneq = statement.g(state.x, 0);
		state.AIneq = statement.g(state.x, 1);
		
		state.active = state.cIneq > -statement.tol;
	
		state.c = [state.c ; state.cIneq(state.active)];
		state.A = [state.A ; state.AIneq(state.active, :)];
	else
		state.cIneq = [];
		state.AIneq = [];
	end
	
	
	
	
	
	
	
	
	
	
	
	
	
	n = length(state.x);
	m = length(state.c);
	state.theta = theta(state.x);
	
	
	% Check if we are done
	if check_stopping_criteria(statement, state)
		plotstate(statement, results, state, constants, false);
		break;
	end
	
	
	
	
	
	
	
	
	
	
	
	
	% Compute search direction
	% If this is too ill-conditioned, we need to perform feasibility restoration
	kktmat = [state.H state.A' ; state.A zeros(m, m)];
	state.condition = cond(kktmat);
	
	if cond(kktmat) > constants.max_condition_number
		disp('This is too ill-conditioned:');
		disp(cond(kktmat));
		
		state.x = restore_feasibility(statement, state.x);
		results.restorations = results.restorations + 1;
		continue;
	end
	% I guess I don't actually have to solve this whole thing (Only solve for d, not lambda)...
	vec = -kktmat \ [state.g ; state.c];
	state.d = vec(1:n);
	
	
	% Backtracking line search
	
	% Start by finding the minimum step length alpha
	if state.g'*state.d < -statement.tol
		state.alpha_min = constants.gamma_alpha * min([constants.gamma_theta, ...
			-constants.gamma_f*state.theta/(state.g'*state.d), ...
			constants.delta*(state.theta)^(constants.s_theta)/(-state.g'*state.d)^(constants.s_f)]);
	else
		state.alpha_min = constants.gamma_alpha * constants.gamma_theta;
	end
	
	
	% Backtrack on alpha
	state.alpha = 1;
	state.accept = false;
	while ~state.accept
		if state.alpha < state.alpha_min
			state.x = restore_feasibility(statement, state.x);
			results.restorations = results.restorations + 1;
			break;
		end
		
		% Calculate new objective/constraint violations
		state.xnew = state.x + state.alpha * state.d;
		state.theta_new = theta(state.xnew);
		state.f_new = statement.f(state.xnew, 0);
		
		plotstate(statement, results, state, constants, true);
		
		% check filter
		if nondom_isdom(results.filter, [state.theta_new; state.f_new])
			state.alpha = state.alpha * (constants.tau_one + constants.tau_two)/2;
			continue;
		end
		
		%m = @(x) (.5*x'*H*x+g'*x+f);
		m = @(a) (a*state.g'*state.d);
		state.ftype = m(state.alpha) < 0 && ...
			((-m(state.alpha))^constants.s_f * state.alpha^(1-constants.s_f) > constants.delta*state.theta^(constants.s_theta));
			
		% Two different accepting criteria
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
		filename = strcat('output/accept_iter_', statement.name,  '_', sprintf('%04d', iteration_number), '.state');
		iteration_number = iteration_number + 1;
		save(filename, 'state');
	
	
		results.ftype_iterations = results.ftype_iterations + 1;
		if state.ftype && (1-constants.gamma_theta)*state.theta_new > statement.tol
			results.filter = nondom_add(results.filter, [(1-constants.gamma_theta)*state.theta_new; state.f-constants.gamma_f*state.theta_new]);
			results.filter_modified_count = results.filter_modified_count + 1;
		end
		
		state.x = state.xnew;
		results.f_min = state.f_new;
		results.x_optimal = state.xnew;
	end
	
	%state
	restults.iters = results.iters + 1;
end





end

