function [gamma_save, Omega_save, adj_save, ar_gamma, info, Y_save] = MCMC_LM_GWishart_probit(X, W, Z, ...
    h_0, h_alpha, h_beta, a, b, lambda, delta_prior, D_prior, ...
    gamma, Omega, burnin, nmc, summary_only)

[n, p] = size(X);
S = X' * X;

% W represents indicators. Initialize Y to random consistent value.
Y = rmvnrnd(zeros(n, 1), eye(n) + h_0 * (ones(n, 1) * ones(1, n)) + h_alpha * (Z * Z'), ...
  1, diag(1 - 2 * W), zeros(n, 1))'; 

% Sampled values of Y to return
Y_save = zeros(n, nmc);

% Additional values as "info" including burnin
full_Y_save = zeros(n, burnin + nmc);

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

% Number of currently included variables
p_gamma = sum(gamma);

% Indicices of currently included variables
ind = find(gamma);

% MCMC sampling
for iter = 1: burnin + nmc
 
    % Print out info every 100 iterations
    if mod(iter, 100) == 0
        fprintf('Iteration = %d\n', iter);
        fprintf('Number of included genes = %d\n', sum(gamma));
        fprintf('Number of add disconnected gene moves proposed %d and accepted %d\n', n_add_disc_prop, n_add_disc_accept);
        fprintf('Number of remove disconnected gene moves proposed %d and accepted %d\n', n_remove_disc_prop, n_remove_disc_accept);
        fprintf('Number of add connectd gene moves proposed %d and accepted %d\n', n_add_conn_prop, n_add_conn_accept);
        fprintf('Number of remove connected gene moves proposed %d and accepted %d\n', n_remove_conn_prop, n_remove_conn_accept);
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
            
            % Prop ratio for MH
            q_ratio = log(p - p_gamma) - log(1 + sum(disconnected_vars));
        else
            % Remove disconnected variable
            ind_dis = find(disconnected_vars);
            
            % Be careful here for case that ind_dis has length 1
            remove_index = ind_dis(randsample(length(ind_dis), 1));
            gamma_prop = gamma;
            gamma_prop(remove_index) = 0;
            n_remove_disc_prop = n_remove_disc_prop + 1;
            
            % Prop ratio for MH
            q_ratio = log(sum(disconnected_vars)) - log(p - p_gamma + 1);
        end
        
        n_gamma_prop = n_gamma_prop + 1;
        
        % Disconnected var, so no update to adjacency
        disconnected = 1;
        adj_prop = adj;
        
        % Compute MH ratio on log scale
        log_r = log_r_y_probit(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta) + ...
            log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected) + ...
            q_ratio;
        
        % Accept proposal with probability r
        if (log(rand(1)) < log_r)
            if (add_var)
                gamma(add_index) = 1;
                p_gamma = p_gamma + 1;
                n_add_disc_accept = n_add_disc_accept + 1;
            else
                gamma(remove_index) = 0;
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
            
            % Prop ratio for MH
            q_ratio = log(p - p_gamma) + log(p_gamma) - log(1 + sum(p1_vars));
            
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
            [Omega_prop] = GWishart_NOij_Gibbs(delta_prior, D_prior(ind_prop, ind_prop), ...
                adj(ind_prop, ind_prop), Omega(ind_prop, ind_prop), i, j, propose_ij, 0, 1);
            G_ratio = log_GWishart_NOij_pdf(delta_prior, D_prior(ind_prop, ind_prop), Omega_prop, i, j, current_ij) - ...
                log_GWishart_NOij_pdf(delta_prior, D_prior(ind_prop, ind_prop), Omega_prop, i, j, propose_ij) - ...
                log_H(delta_prior, D_prior(ind_prop, ind_prop), n, S(ind_prop, ind_prop), Omega(ind_prop, ind_prop), i, j);
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
            [Omega_prop] = GWishart_NOij_Gibbs(delta_prior, D_prior(ind, ind), ...
                adj(ind, ind), Omega(ind, ind), i, j, propose_ij, 0, 1);
            G_ratio = log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop, i, j, current_ij) - ...
                log_GWishart_NOij_pdf(delta_prior, D_prior(ind, ind), Omega_prop, i, j, propose_ij) + ...
                log_H(delta_prior, D_prior(ind, ind), n, S(ind, ind), Omega(ind, ind), i, j);
        end
        
        n_gamma_prop = n_gamma_prop + 1;
        disconnected = 0;
        
        % Compute MH ratio on log scale
        log_r = log_r_y_probit(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta) + ...
            log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected) + ...
            q_ratio + G_ratio;
        
        % Accept proposal with probability r
        if (log(rand(1)) < log_r)
            if (add_var)
                gamma(add_index) = 1;
                p_gamma = p_gamma + 1;
                n_add_conn_accept = n_add_conn_accept + 1;
            else
                gamma(remove_index) = 0;
                p_gamma = p_gamma - 1;
                n_remove_conn_accept = n_remove_conn_accept + 1;
                
                % Zero out corresponding elements of Omega
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
    
    %  Update Omega_gamma given graph and fill in adj
    [Omega(ind, ind)] = GWishart_BIPS_maximumClique(delta_post, ...
        D_post(ind, ind), adj_gamma, Omega(ind, ind), 0, 1);
    [adj(ind, ind)] = adj_gamma;
    
    % Update remaining entries of Omega
    for i = 1:p
        if ~gamma(i)
            % Note that parameter scale = 1/rate
            Omega(i, i) = gamrnd(0.5 * delta_post, 2 / D_post(i, i));
        end
    end
    
    X_gamma = X(:, find(gamma));
    Sigma = eye(n) + h_0 * (ones(n, 1) * ones(1, n)) + h_alpha * (Z * Z') + h_beta * (X_gamma * X_gamma');
    mu = zeros(n, 1);
    Y = rmvnrnd(mu, Sigma, 1, diag(1 - 2 * W), zeros(n, 1))'; 
    
    if iter > burnin
        gamma_save(:, iter-burnin) = gamma;
        Y_save(:, iter-burnin) = Y;

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
    full_Y_save(:, iter) = Y;

end

ar_gamma = n_gamma_accept / n_gamma_prop;

% Info for diagnostic purposes
info = struct('n_add_disc_prop', n_add_disc_prop, 'n_add_disc_accept', n_add_disc_accept, ...
    'n_remove_disc_prop', n_remove_disc_prop, 'n_remove_disc_accept', n_remove_disc_accept, ...
    'n_add_conn_prop', n_add_conn_prop, 'n_add_conn_accept', n_add_conn_accept, ...
    'n_remove_conn_prop', n_remove_conn_prop, 'n_remove_conn_accept', n_remove_conn_accept, ...
    'n_no_prop', n_no_prop, ...
    'full_gamma', full_gamma_save, ...
    'node_degrees', node_degrees, ...
    'full_Y', full_Y_save);