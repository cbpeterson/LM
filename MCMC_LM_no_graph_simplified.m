function [gamma_save] = MCMC_LM_no_graph_simplified(X, Y, Z, ...
    a_0, b_0, h_alpha, h_beta, lambda, gamma, burnin, nmc)

[n, p] = size(X);

% Allocate storage for MCMC sample
gamma_save = zeros(p, nmc);

% Number of currently included variables
p_gamma = sum(gamma);

% Intercept term not included here
h_0 = 0;

% MCMC sampling
for iter = 1: burnin + nmc
    
    % Add/remove variable with prob 1/2
    if (binornd(1, 0.5))
        % Add variable
        % Select variable to add from those not current included
        if (p_gamma < p)
            ind_not = find(gamma == 0);
            
            % Need to be careful here for when ind_not has length 1
            add_index = ind_not(randsample(length(ind_not), 1));
            gamma_prop = gamma;
            gamma_prop(add_index) = 1;
            
            % Compute MH ratio on log scale
            log_r = log_r_y(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta, a_0, b_0) + ...
                log(lambda) - log(1 - lambda) + ...
                log(p - p_gamma) - log(1 + p_gamma);
            
            % Accept proposal with probability r
            if (log(rand(1)) < log_r)
                gamma(add_index) = 1;
                p_gamma = p_gamma + 1;
            end
        end
        
    else
        % Remove variable at random from those currently included
        if (sum(gamma) > 0)
            ind_incl = find(gamma);
            
            % Be careful here for case that ind_dis has length 1
            remove_index = ind_incl(randsample(length(ind_incl), 1));
            gamma_prop = gamma;
            gamma_prop(remove_index) = 0;
            
            % Compute MH ratio on log scale
            log_r = log_r_y(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta, a_0, b_0) + ...
                log(1 - lambda) - log(lambda) + ...
                log(p_gamma) - log(p - p_gamma + 1);
            
            % Accept proposal with probability r
            if (log(rand(1)) < log_r)
                gamma(remove_index) = 0;
                p_gamma = p_gamma - 1;
            end
        end
    end

    if iter > burnin
        gamma_save(:, iter-burnin) = gamma; 
    end

end

