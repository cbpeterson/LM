function [gamma_save, Omega_save, adj_save, ar_gamma, info] = MCMC_LM_GWishart_simplified(X, Y, Z, ...
    a_0, b_0, h_alpha, h_beta, a, b, lambda, delta_prior, D_prior, ...
    gamma, Omega, burnin, nmc, summary_only)

[n, p] = size(X);
S = X' * X;

% Posterior parameters for G-Wishart
delta_post = delta_prior + n;
D_post = D_prior + S;

% Always keep variable selections
gamma_save = zeros(p, nmc);

% Record some diagnostic info
n_add_prop_save = zeros(1, burnin + nmc);
n_add_accept_save = zeros(1, burnin + nmc);
n_remove_prop_save = zeros(1, burnin + nmc);
n_remove_accept_save = zeros(1, burnin + nmc);
n_no_prop_save = zeros(1, burnin + nmc);
full_gamma_save = zeros(p, burnin + nmc);
node_degrees = zeros(p, burnin + nmc);

% Keep track of info to compute acceptance rates
n_gamma_prop = 0;
n_gamma_accept = 0;
n_add_prop = 0;
n_add_accept = 0;
n_remove_prop = 0;
n_remove_accept = 0;
n_no_prop = 0;

% Allocate storage for MCMC sample, or just for means if only summary is
% required
if summary_only
    Omega_save = zeros(p, p);
    adj_save = Omega_save;
else
    Omega_save = zeros(p, p, nmc);
    adj_save = Omega_save;
end

% Initial value of adjacency matrix
adj = abs(Omega) > 1e-5;

% Zero out elements of Omega close to 0
Omega = Omega .* adj;

% Number of currently included variables
p_gamma = sum(gamma);

% Indicices of currently included variables
ind = find(gamma);

p_add = 0.5;

% MCMC sampling
for iter = 1: burnin + nmc
 
    % For large p (when things may be slow), print out 100 iterations
    if p > 200 && mod(iter, 100) == 0
        fprintf('padd = %.2f\n', p_add);
        fprintf('Iteration = %d\n', iter);
        fprintf('Number of included genes = %d\n', sum(gamma));
        fprintf('Number of add gene moves proposed %d and accepted %d\n', n_add_prop, n_add_accept);
        fprintf('Number of remove gene moves proposed %d and accepted %d\n', n_remove_prop, n_remove_accept);
        fprintf('Number of times we could not propose removal %d \n', n_no_prop);
        fprintf('Number of included edges %d \n\n', (sum(sum(adj)) - p) / 2);
    end
    
    % Add/remove variable with prob 1/2
    % INSTEAD, MAKE SURE WE ARE PROPOSING APPROX EQUALLY NUMBERS OF ADD AND
    % REMOVES
    if (binornd(1, p_add))
        % Add variable
        % Select variable to add from those not current included
        if (p_gamma < p)
            ind_not = find(gamma == 0);

            % Need to be careful here for when ind_not has length 1
            add_index = ind_not(randsample(length(ind_not), 1));
            gamma_prop = gamma;
            gamma_prop(add_index) = 1;
            n_gamma_prop = n_gamma_prop + 1;
            n_add_prop = n_add_prop + 1;
            
            % Simple version where we add a disconnected var
            disconnected = 1;
            adj_prop = adj;
            
            % Compute MH ratio on log scale
            log_r = log_r_y(gamma, gamma_prop, X, Y, Z, h_alpha, h_beta, a_0, b_0) + ...
                log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected) + ...
                log(p - p_gamma) - log(1 + p_gamma);
            
            % Accept proposal with probability r
            if (log(rand(1)) < log_r)
                gamma(add_index) = 1;
                p_gamma = p_gamma + 1;
                ind = find(gamma);
                n_gamma_accept = n_gamma_accept + 1;
                n_add_accept = n_add_accept + 1;
            end
        end
    else
        % Remove variable
        degree = sum(adj, 2) - 1;
        disconnected_vars = (degree == 0) & gamma;
        p1_vars = (degree == 1) & gamma;
        
        % Propose removing a variable connected by one edge 1/2 of the
        % time, as long as there is one
        if (sum(p1_vars) == 0 || (sum(disconnected_vars) > 0 && binornd(1, 0.5)))
            % Remove a disconnected variable
            if (sum(disconnected_vars) > 0)
                ind_dis = find(disconnected_vars);
                
                % Be careful here for case that ind_dis has length 1
                remove_index = ind_dis(randsample(length(ind_dis), 1));
                gamma_prop = gamma;
                gamma_prop(remove_index) = 0;
                n_gamma_prop = n_gamma_prop + 1;
                n_remove_prop = n_remove_prop + 1;
                
                % Simple version where we remove a disconnected var
                disconnected = 1;
                adj_prop = adj;
                
                % Compute MH ratio on log scale
                log_r = log_r_y(gamma, gamma_prop, X, Y, Z, h_alpha, h_beta, a_0, b_0) + ...
                    log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected) + ...
                    log(sum(disconnected_vars)) - log(p - p_gamma + 1);
                
                % Accept proposal with probability r
                if (log(rand(1)) < log_r)
                    gamma(remove_index) = 0;
                    p_gamma = p_gamma - 1;
                    ind = find(gamma);
                    n_gamma_accept = n_gamma_accept + 1;
                    n_remove_accept = n_remove_accept + 1;
                end
            else
                n_no_prop = n_no_prop + 1;
            end
        else
            % Remove a variable connected by a single edge 
            if (sum(p1_vars) > 0)
                ind_p1 = find(p1_vars);
                
                % Be careful here for case that ind_p1 has length 1
                remove_index = ind_p1(randsample(length(ind_p1), 1));
                gamma_prop = gamma;
                gamma_prop(remove_index) = 0;
                n_gamma_prop = n_gamma_prop + 1;
                n_remove_prop = n_remove_prop + 1;
                
                % Now need to deal with updates to G and Omega
                disconnected = 0;
                edge_ind = setdiff(find(adj(remove_index, :)), remove_index);
                adj_prop = adj;
                adj_prop(remove_index, edge_ind) = 0;
                adj_prop(edge_ind, remove_index) = 0;
                
                % Step 1(b) - generate proposal for Omega
                % Treat Omega and Omega_prop as being in dimension p_gamma
                % Requires fixing up i and j
                change_inds = zeros(p, 1);
                change_inds(remove_index) = 1;
                change_inds(edge_ind) = 1;
                change_inds_gamma = change_inds(ind);
                i = find(change_inds_gamma, 1);
                j = find(change_inds_gamma, 1, 'last');
                current_ij = 1;
                propose_ij = 0;
                try
                    [Omega_prop] = GWishart_NOij_Gibbs(delta_prior, D_prior(ind, ind), ...
                        adj(ind, ind), Omega(ind, ind), i, j, propose_ij, 0, 1);
                catch
                    error('Error thrown from GWishart_NOij_Gibbs in step 1(b) when removing a connected var');
                end
                
                % Compute MH ratio on log scale
                log_r = log_r_y(gamma, gamma_prop, X, Y, Z, h_alpha, h_beta, a_0, b_0) + ...
                    log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected) + ...
                    log(sum(p1_vars)) - log(p - p_gamma + 1) - log(p_gamma - 1) + ...
                    log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop, i, j, current_ij) - ...
                    log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop, i, j, propose_ij) + ...
                    log_H(delta_prior, D_prior(ind, ind), n, S(ind, ind), Omega(ind, ind), i, j);
                
                % Accept proposal with probability r
                if (log(rand(1)) < log_r)
                    gamma(remove_index) = 0;
                    p_gamma = p_gamma - 1;
                    adj = adj_prop;
                    current_ij = propose_ij;
                    n_gamma_accept = n_gamma_accept + 1;
                    n_remove_accept = n_remove_accept + 1;
                    
                    % Zero out corresponding elements of Omega
                    Omega(remove_index, edge_ind) = 0;
                    Omega(edge_ind, remove_index) = 0;
                end
                
                % Step 2(c)
                % adj has been updated, but ind has not
                try
                [Omega(ind, ind)] = GWishart_NOij_Gibbs(delta_post, D_post(ind, ind), ...
                    adj(ind, ind), Omega(ind, ind), i, j, current_ij, 0, 0);
                catch
                    error('Error thrown from GWishart_NOij_Gibbs in step 2(c) when removing a connected var');
                end
                
                % Update ind now to avoid messing up indices of i and j
                ind = find(gamma);
                
                %  Update Omega_gamma given graph
                [Omega(ind, ind)] = GWishart_BIPS_maximumClique(delta_post, ...
                    D_post(ind, ind), adj(ind, ind), Omega(ind, ind), 0, 1); 
            else
                n_no_prop = n_no_prop + 1;
            end
        end
    end
    
    % Update p_add to ensure balance of proposals
    if (n_add_prop + n_remove_prop > 0)
       p_add = (n_remove_prop) / (n_add_prop + n_remove_prop);
    end
    
    % Error check that all off-diagonal elements are in fact 0
    not_ind = setdiff(1:p, ind);
    if ((size(find(Omega(not_ind, not_ind) - diag(diag(Omega(not_ind, not_ind)))), 1) ~= 0) || ...
            (size(find(Omega(ind, not_ind)), 1) ~= 0))
        error('Not all off-diagonal elements that should be zero are zero');
    end
    
    % Sample graph and precision matrix for a given set of variables
    adj_gamma = adj(ind, ind);
    
    % Sample off-diagonal elements corresponding to included variables
    for i = 1:(p_gamma - 1) % modified to only cycle over included vars
        for j = (i + 1):p_gamma
            % Step 1(a) following modification at the bottom of p. 186
            % Bernoulli proposal with odds p(G') * H(e, Sigma) / P(G)
            % Log of odds
            w = log_H(delta_prior, D_prior(ind, ind), n, S(ind, ind), Omega(ind, ind), i, j) -  ...
                2 * b + log(1 - lambda) - log(lambda);
            
            % 1 / (exp(log odds) + 1) = 1 - Bernoulli probability
            w = 1 / (exp(w) + 1);

            % current_ij indicates whether edge (i,j) is in G
            current_ij = adj_gamma(i, j);
            
            % propose_ij indicates whether edge (i,j) is in G'
            % Proposal will be 1 if rand(1) < w i.e. will be an edge
            % with probability w
            propose_ij = rand(1) < w;
            if (propose_ij ~= current_ij)
                % Step 1(b)
                try
                [Omega_prop] = GWishart_NOij_Gibbs(delta_prior, D_prior(ind, ind), adj_gamma, ...
                    Omega(ind, ind), i, j, propose_ij, 0, 1);
                catch
                    nrep = size(ind, 1);
                    ind_str = '';
                    for ind_idx = 1:nrep
                        ind_str = [ind_str, ', ', num2str(ind(ind_idx, 1))];
                    end
                    error('Error at step 1b with param values\n ind = %s\n i = %d\n j = %d', ind_str, i, j);
                end
                
                % Step 2(b) from expression at top of p. 189
                r2 = log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop, i, j, current_ij) ...
                    - log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop, i, j, propose_ij);
                
                % Acceptance rate alpha = min(1, e^r2)
                if (log(rand(1)) < r2) 
                    adj_gamma(i, j) = propose_ij;
                    adj_gamma(j, i) = propose_ij;
                    current_ij = propose_ij;
                end      
            end
            
            % Step 2(c)
            try
            [Omega(ind, ind)] = GWishart_NOij_Gibbs(delta_post, D_post(ind, ind), ...
                adj_gamma, Omega(ind, ind), i, j, current_ij, 0, 0);
            catch
                nrep = size(ind, 1);
                ind_str = '';
                for ind_idx = 1:nrep
                    ind_str = [ind_str, ', ', num2str(ind(ind_idx, 1))];
                end
                error('Error at step 2c param values\n ind = %s\n i = %d\n j = %d', ind_str, i, j);
            end
            
            % if (min(eig(Omega)) < 0)
            %   error('pos 4.1');
            % end
        end
    end
    
    %  Update Omega_gamma given graph and fill in adj
    [Omega(ind, ind)] = GWishart_BIPS_maximumClique(delta_post, ...
        D_post(ind, ind), adj_gamma, Omega(ind, ind), 0, 1);
    [adj(ind, ind)] = adj_gamma;
    
    % Update remaining entries of Omega
    for i = 1:p
        if ~gamma(i)
            % Note that parameter scale = 1/rate
            Omega(i, i) = gamrnd(0.5 * delta_post, 2 / D_post(i, i));
            
            % if (min(eig(Omega)) < 0)
            %    error('Neg eignvalue after updating diagonal elements');
            % end
        end
    end
    
    % Check that adj and Omega are consistent
    % if (sum(sum((Omega ~= 0) ~= adj)) ~= 0)
    %     error('Omega and adj are not consistent at end of MCMC iteration')
    % end
    
    % At this point we need for Omega to be pos def
    % if (min(eig(Omega)) < 0)
    %    error('Omega not pos def at end of iteration');
    % end
    
    if iter > burnin
        gamma_save(:, iter-burnin) = gamma;

        if summary_only
            Omega_save(:, :) = Omega_save(:, :) + Omega / nmc;
            adj_save(:, :) = adj_save(:, :) + adj / nmc;
        else
            Omega_save(:, :, iter-burnin) = Omega;
            adj_save(:, :, iter-burnin) = adj;
        end
    end
    
    n_add_prop_save(1, iter) = n_add_prop;
    n_add_accept_save(1, iter) = n_add_accept;
    n_remove_prop_save(1, iter) = n_remove_prop;
    n_remove_accept_save(1, iter) = n_remove_accept;
    n_no_prop_save(1, iter) = n_no_prop;
    full_gamma_save(:, iter) = gamma;
    node_degrees(:, iter) = sum(adj, 2) - 1;

end

ar_gamma = n_gamma_accept / n_gamma_prop;

% Info for diagnostic purposes
info = struct('n_add_prop', n_add_prop_save, 'n_add_accept', n_add_accept_save, ...
    'n_remove_prop', n_remove_prop_save, 'n_remove_accept', n_remove_accept_save, ...
    'n_no_prop', n_no_prop_save, 'full_gamma', full_gamma_save, ...
    'node_degrees', node_degrees);