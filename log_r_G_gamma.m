function [log_MH] = log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected, ...
  omega_ii, delta, d_ii)
% Compute MH ratio for adding or removing one var

p = size(gamma, 1);

% +1 if adding, -1 if removing a var
gamma_diff = sum(gamma_prop - gamma);

% Disconnected = true if no edges are changed
if (disconnected)    
    % Smaller number of included vars
    min_vars = min(sum(gamma_prop), sum(gamma));
    
    log_MH = gamma_diff * (log(gampdf(omega_ii, 0.5 * delta, 2 / d_ii)) + ...
        a + min_vars * log(1 - lambda)) + log(calc_C(sum(gamma), b, lambda)) - ...
        log(calc_C(sum(gamma_prop), b, lambda));
else
    % Assumption in paper is that adjacency matrix has 0's along the diagonal,
    % while here is has 1's, so need to subtract eye(p)
    adj = adj - eye(p);
    adj_prop = adj_prop - eye(p);
    
    % Index of added/removing var
    i = find(gamma ~= gamma_prop);

    if (gamma_diff == 1)
        % Adding a variable
        inds = find(gamma);
        edges_i = adj_prop(i, inds);
    else
        % Removing a variable
        inds = find(gamma_prop);
        edges_i = adj(i, inds);
    end
    
    num_edges = sum(edges_i);
    num_missing = sum(edges_i == 0);
    
    log_MH = gamma_diff * (log(gampdf(omega_ii, 0.5 * delta, 2 / d_ii)) + a) + ...
        log(calc_C(sum(gamma), b, lambda)) - ...
        log(calc_C(sum(gamma_prop), b, lambda)) + ...
        b * (gamma_prop' * adj_prop * gamma_prop - ...
        gamma' * adj * gamma) + ...
        gamma_diff * (num_edges * log(lambda) + num_missing * log(1-lambda));
end