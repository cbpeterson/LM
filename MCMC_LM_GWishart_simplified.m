function [gamma_save, Omega_save, adj_save, ar_gamma, info] = MCMC_LM_GWishart_simplified(X, Y, Z, ...
    a_0, b_0, h_alpha, h_beta, a, b, lambda, delta_prior, D_prior, h_gamma, k_0, delta_c, ...
    gamma, Omega, burnin, nmc, summary_only)

[n, p] = size(X);
S = X' * X;

% Posterior parameters for G-Wishart
delta_post = delta_prior + n;
D_post = D_prior + S;

% Always keep variable selections
gamma_save = zeros(p, nmc);

% Record some diagnostic info
full_gamma_save = zeros(p, burnin + nmc);
node_degrees = zeros(p, burnin + nmc);

% Keep track of info to compute acceptance rates
n_gamma_prop = 0; n_gamma_accept = 0;
n_add_disc_prop = 0; n_add_disc_accept = 0;
n_remove_disc_prop = 0; n_remove_disc_accept = 0;
n_add_conn_prop = 0; n_add_conn_accept = 0;
n_remove_conn_prop = 0; n_remove_conn_accept = 0;
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

% Initial version of Sigma
Sig = inv(Omega);

% Number of currently included variables
p_gamma = sum(gamma);

% Indicices of currently included variables
ind = find(gamma);

% No intercept term included here
h_0 = 0;

% MCMC sampling
for iter = 1: burnin + nmc
 
    % Print out info every 100 iterations
    if mod(iter, 100) == 0
        fprintf('Iteration = %d\n', iter);
        fprintf('Number of included variables = %d\n', sum(gamma));
        fprintf('Number of add disconnected variable moves proposed %d and accepted %d\n', n_add_disc_prop, n_add_disc_accept);
        fprintf('Number of remove disconnected variable moves proposed %d and accepted %d\n', n_remove_disc_prop, n_remove_disc_accept);
        fprintf('Number of add connected variable moves proposed %d and accepted %d\n', n_add_conn_prop, n_add_conn_accept);
        fprintf('Number of remove connected variable moves proposed %d and accepted %d\n', n_remove_conn_prop, n_remove_conn_accept);
        fprintf('Number of times we could not make a proposal %d \n', n_no_prop);
        fprintf('Number of included edges %d \n\n', (sum(sum(adj)) - p) / 2);
    end
    
    % Propose add/remove of disconnected vs connected var with prop 1/2
    degree = sum(adj, 2) - 1;
    disconnected_vars = (degree == 0) & gamma;
    p1_vars = (degree == 1) & gamma;
    
    disc_possible = ((p_gamma < p) && (sum(disconnected_vars) > 0)) || (p_gamma == 0);
    conn_possible = (p_gamma < p) && (sum(p1_vars) > 0);
    if (disc_possible && conn_possible)
        disc_prop = binornd(1, 0.5);
        conn_prop = ~disc_prop;
    elseif (disc_possible)
        disc_prop = 1;
        conn_prop = 0;
    elseif (conn_possible)
        disc_prop = 0;
        conn_prop = 1;
    else
        disc_prop = 0;
        conn_prop = 0;
        n_no_prop = n_no_prop + 1;
    end
    
    if (disc_prop)
        % Disconnected variable
        if (p_gamma > 0)
          add_var = binornd(1, 0.5);
        else
           add_var = 1;
        end
        
        if (add_var)
            % Add disconnected variable
            ind_not = find(gamma == 0);
            
            % Need to be careful here for when ind_not has length 1
            add_index = ind_not(randsample(length(ind_not), 1));
            gamma_prop = gamma;
            gamma_prop(add_index) = 1;
            n_add_disc_prop = n_add_disc_prop + 1;
            
            % Need to propose new diagonal entry for Omega
            omega_add_index_prop = gamrnd(0.5 * delta_post, 2 / D_post(add_index, add_index));
            Omega_prop = Omega;
            Omega_prop(add_index, add_index) = omega_add_index_prop;
            omega_ii = Omega_prop(add_index, add_index);
            d_ii = D_prior(add_index, add_index);
            
            % Prop ratio for MH
            q_ratio = log(p - p_gamma) - log(1 + sum(disconnected_vars)) - ...
                log(gampdf(omega_add_index_prop, 0.5 * delta_post, 2 / D_post(add_index, add_index)));
        else
            % Remove disconnected variable
            ind_dis = find(disconnected_vars);
            
            % Be careful here for case that ind_dis has length 1
            remove_index = ind_dis(randsample(length(ind_dis), 1));
            gamma_prop = gamma;
            gamma_prop(remove_index) = 0;
            n_remove_disc_prop = n_remove_disc_prop + 1;
            Omega_prop = Omega;
            Omega_prop(remove_index, remove_index) = 0;
            omega_ii = Omega(remove_index, remove_index);
            d_ii = D_prior(remove_index, remove_index);
            
            % Prop ratio for MH
            q_ratio = log(sum(disconnected_vars)) - log(p - p_gamma + 1) + ...
                log(gampdf(Omega(remove_index, remove_index), 0.5 * delta_post, 2 / D_post(add_index, add_index)));
        end
        
        n_gamma_prop = n_gamma_prop + 1;
        
        % Disconnected var, so no update to adjacency
        disconnected = 1;
        adj_prop = adj;
        
        % Compute MH ratio on log scale
        log_r = log_r_y(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta, a_0, b_0) + ...
            log_r_X(gamma, gamma_prop, X, Omega, Omega_prop, h_gamma, k_0, delta_c) + ...
            log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected, ...
              omega_ii, delta_prior, d_ii) + ...
            q_ratio;
        
        % Accept proposal with probability r
        if (log(rand(1)) < log_r)
            if (add_var)
                gamma(add_index) = 1;
                Omega = Omega_prop;
                p_gamma = p_gamma + 1;
                n_add_disc_accept = n_add_disc_accept + 1;
            else
                gamma(remove_index) = 0;
                Omega = Omega_prop;
                p_gamma = p_gamma - 1;
                n_remove_disc_accept = n_remove_disc_accept + 1;
            end
            ind = find(gamma);
            n_gamma_accept = n_gamma_accept + 1;
        end
    elseif (conn_prop)
        % Connected variable
        add_var = binornd(1, 0.5);
        
        if (add_var)
            % Add variable connected by one edge
            ind_not = find(gamma == 0);
            
            % Need to be careful here for when ind_not has length 1
            add_index = ind_not(randsample(length(ind_not), 1));
            gamma_prop = gamma;
            gamma_prop(add_index) = 1;
            ind_prop = find(gamma_prop);
            n_add_conn_prop = n_add_conn_prop + 1;
            
            % Need to propose new diagonal entry for Omega
            omega_add_index_prop = gamrnd(0.5 * delta_post, 2 / D_post(add_index, add_index));
            Omega_prop = Omega;
            Omega_prop(add_index, add_index) = omega_add_index_prop;
            omega_ii = Omega_prop(add_index, add_index);
            d_ii = D_prior(add_index, add_index);
            
            % Need to make sure diagonal entry of Omega used below in log_H is
            % nonzero
            Omega_log_H = Omega;
            Omega_log_H(add_index, add_index) = omega_add_index_prop;
            
            % Prop ratio for MH
            q_ratio = log(p - p_gamma) + log(p_gamma) - log(1 + sum(p1_vars)) - ...
              log(gampdf(omega_add_index_prop, 0.5 * delta_post, 2 / D_post(add_index, add_index)));
            
            % Now need to deal with updates to G and Omega
            % Pick a currently included variable
            edge_ind = ind(randsample(length(ind), 1));
            adj_prop = adj;
            adj_prop(add_index, edge_ind) = 1;
            adj_prop(edge_ind, add_index) = 1;
            
            % Step 1(b) - generate proposal for Omega
            % Treat Omega and Omega_prop as being in dimension p_gamma + 1
            % Requires fixing up i and j
            change_inds = zeros(p, 1);
            change_inds(add_index) = 1;
            change_inds(edge_ind) = 1;
            change_inds_gamma_prop = change_inds(ind_prop);
            i = find(change_inds_gamma_prop, 1);
            j = find(change_inds_gamma_prop, 1, 'last');
            current_ij = 0;
            propose_ij = 1;
            Omega_prop(ind_prop, ind_prop) = GWishart_NOij_Gibbs(delta_prior, ...
                D_prior(ind_prop, ind_prop), ...
                adj(ind_prop, ind_prop), Omega_prop(ind_prop, ind_prop), i, j, propose_ij, 0, 1);
            G_ratio = log_GWishart_NOij_pdf(delta_prior, D_prior(ind_prop, ind_prop), ...
                Omega_prop(ind_prop, ind_prop), i, j, current_ij) - ...
                log_GWishart_NOij_pdf(delta_prior, D_prior(ind_prop, ind_prop), ...
                Omega_prop(ind_prop, ind_prop), i, j, propose_ij) - ...
                log_H(delta_prior, D_prior(ind_prop, ind_prop), n, S(ind_prop, ind_prop), ...
                  Omega_log_H(ind_prop, ind_prop), i, j);
        else
            % Remove variable connected by one edge
            ind_p1 = find(p1_vars);
            
            % Be careful here for case that ind_p1 has length 1
            remove_index = ind_p1(randsample(length(ind_p1), 1));
            gamma_prop = gamma;
            gamma_prop(remove_index) = 0;
            n_remove_conn_prop = n_remove_conn_prop + 1;
            
            % Prop ratio for MH
            q_ratio = log(sum(p1_vars)) - log(p - p_gamma + 1) - log(p_gamma - 1);
            
            % Make sure precision matrix is updated
            Omega_prop = Omega;
            Omega_prop(remove_index, remove_index) = 0;
            omega_ii = Omega(remove_index, remove_index);
            d_ii = D_prior(remove_index, remove_index);
            
            % Now need to deal with updates to G and Omega
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
            Omega_prop(ind, ind) = GWishart_NOij_Gibbs(delta_prior, D_prior(ind, ind), ...
                adj(ind, ind), Omega(ind, ind), i, j, propose_ij, 0, 1);
            G_ratio = log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop(ind, ind), i, j, current_ij) - ...
                log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop(ind, ind), i, j, propose_ij) + ...
                log_H(delta_prior, D_prior(ind, ind), n, S(ind, ind), Omega(ind, ind), i, j);
        end
        
        n_gamma_prop = n_gamma_prop + 1;
        disconnected = 0;
        
        % Compute MH ratio on log scale
        log_r = log_r_y(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta, a_0, b_0) + ...
            log_r_X(gamma, gamma_prop, X, Omega, Omega_prop, h_gamma, k_0, delta_c) + ...
            log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected, ...
              omega_ii, delta_prior, d_ii) + ...
            q_ratio + G_ratio;
        
        % Accept proposal with probability r
        if (log(rand(1)) < log_r)
            if (add_var)
                gamma(add_index) = 1;
                p_gamma = p_gamma + 1;
                n_add_conn_accept = n_add_conn_accept + 1;
                
                % Set diagonal entry of Omega to be nonzero
                Omega(add_index, add_index) = Omega_prop(add_index, add_index);
            else
                gamma(remove_index) = 0;
                p_gamma = p_gamma - 1;
                n_remove_conn_accept = n_remove_conn_accept + 1;
                
                % Zero out corresponding elements of Omega
                Omega(remove_index, remove_index) = 0;
                Omega(remove_index, edge_ind) = 0;
                Omega(edge_ind, remove_index) = 0;
            end
            
            adj = adj_prop;
            current_ij = propose_ij;
            n_gamma_accept = n_gamma_accept + 1;
            
            if (add_var)
                % Step 2(c)
                % adj has been updated, but ind has not
                [Omega(ind_prop, ind_prop)] = GWishart_NOij_Gibbs(delta_post, D_post(ind_prop, ind_prop), ...
                    adj(ind_prop, ind_prop), Omega(ind_prop, ind_prop), i, j, current_ij, 0, 0);
            else
                % Step 2(c)
                % adj has been updated, but ind has not
                [Omega(ind, ind)] = GWishart_NOij_Gibbs(delta_post, D_post(ind, ind), ...
                    adj(ind, ind), Omega(ind, ind), i, j, current_ij, 0, 0);
            end
            
            % Update ind now to avoid messing up indices of i and j
            ind = find(gamma);
            
            %  Update Omega_gamma given graph
            [Omega(ind, ind)] = GWishart_BIPS_maximumClique(delta_post, ...
                D_post(ind, ind), adj(ind, ind), Omega(ind, ind), 0, 1);
        end
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
        end
    end
    
    %  Update Omega_gamma given graph and fill in adj. Also update Sig
    Sig = zeros(p);
    [Omega(ind, ind), Sig(ind, ind)] = GWishart_BIPS_maximumClique(delta_post, ...
        D_post(ind, ind), adj_gamma, Omega(ind, ind), 0, 1);
    [adj(ind, ind)] = adj_gamma;
    
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
    
    full_gamma_save(:, iter) = gamma;
    node_degrees(:, iter) = sum(adj, 2) - 1;

end

ar_gamma = n_gamma_accept / n_gamma_prop;

% Info for diagnostic purposes
info = struct('n_add_disc_prop', n_add_disc_prop, 'n_add_disc_accept', n_add_disc_accept, ...
    'n_remove_disc_prop', n_remove_disc_prop, 'n_remove_disc_accept', n_remove_disc_accept, ...
    'n_add_conn_prop', n_add_conn_prop, 'n_add_conn_accept', n_add_conn_accept, ...
    'n_remove_conn_prop', n_remove_conn_prop, 'n_remove_conn_accept', n_remove_conn_accept, ...
    'n_no_prop', n_no_prop, ...
    'full_gamma', full_gamma_save, ...
    'node_degrees', node_degrees);