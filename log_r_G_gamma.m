function [log_MH] = log_r_G_gamma(gamma, gamma_prop, adj, adj_prop, a, b, lambda, disconnected)
% Compute MH ratio for adding or removing one var

p = size(gamma, 1);

% +1 if adding, -1 if removing a var
gamma_diff = sum(gamma_prop - gamma);

% Disconnected = true if no edges are changed
if (disconnected)    
    % Smaller number of included vars
    min_vars = min(sum(gamma_prop), sum(gamma));
    
    log_MH = gamma_diff * (a + min_vars * log(1 - lambda));
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
    
    log_MH = gamma_diff * a + ...
        b * (gamma_prop' * adj_prop * gamma_prop - ...
        gamma' * adj * gamma) + ...
        gamma_diff * (num_edges * log(lambda) + num_missing * log(1-lambda));
    
    % Temp - compute full ratio including all terms, should match result above
    numer_prod = 0;
    denom_prod = 0;
    for j = 2:p
        for k = 1:(j-1)
            numer_prod = numer_prod + gamma_prop(j) * gamma_prop(k) * ...
                (adj_prop(j, k) * log(lambda) + (1 - adj_prop(j, k)) * log(1-lambda));
            denom_prod = denom_prod + gamma(j) * gamma(k) * ...
                (adj(j, k) * log(lambda) + (1 - adj(j, k)) * log(1-lambda));
        end
    end
    
    log_MH_check = a * ones(1, p) * gamma_prop + b * gamma_prop' * adj_prop * gamma_prop + ...
        numer_prod - a * ones(1, p) * gamma - b * gamma' * adj * gamma - denom_prod;
    
    if (abs(log_MH - log_MH_check) > 1e-5)
        error('log_MH_r_Gamma', 'log MH r Gamma values do not match');
    end    
end

end