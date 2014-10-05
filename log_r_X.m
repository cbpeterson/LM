function [log_X_mh_ratio] = log_r_X(gamma, gamma_prop, X, Omega, Omega_prop, h_gamma, k_0, delta_c)
  ind = find(gamma);
  ind_prop = find(gamma_prop);
  ind_not = find(gamma == 0);
  ind_not_prop = find(gamma_prop == 0);
  
  % Compute MH ratio on log scale
  log_X_mh_ratio = log_X_included(X(:, ind_prop), Omega_prop(ind_prop, ind_prop)) + ...
      log_X_not_included(X(:, ind_not_prop), X(:, ind_prop), h_gamma, k_0, delta_c) - ...
      log_X_included(X(:, ind), Omega(ind, ind)) - ...
      log_X_not_included(X(:, ind_not), X(:, ind), h_gamma, k_0, delta_c);
end

