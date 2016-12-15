function [constants] = create_constants(statement)

constants = struct();
constants.theta_max   = 50 * statement.theta(statement.x0); % (theta(x0), infty)
constants.gamma_theta = .01; % (0,1)
constants.gamma_f     = .75; % (0,1)
constants.delta       = .01; % (0,infty)
constants.gamma_alpha = .5; % (0,1]
constants.s_theta     = 2; % (1,infty)
constants.s_f         = 3; % [1,infty)
constants.eta_f       = .25; % (0, .5)
constants.tau_one     = .25; % (0, tau_two]
constants.tau_two     = .75; % [tau_two, 1)


end
