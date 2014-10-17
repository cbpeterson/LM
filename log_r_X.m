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
  
  % Try comparing this to MH ratio assuming that p_X is matrix variate t
   log_X_mh_ratio2 = log_X_included(X(:, ind_prop), Omega_prop(ind_prop, ind_prop)) + ...
      log_X_not_included_rev(X(:, ind_not_prop), X(:, ind_prop), h_gamma, k_0, delta_c) - ...
      log_X_included(X(:, ind), Omega(ind, ind)) - ...
      log_X_not_included_rev(X(:, ind_not), X(:, ind), h_gamma, k_0, delta_c);
  % As it turns out, this ratio is the same, so distr really is a matrix t
  
   if (abs(log_X_mh_ratio - log_X_mh_ratio2) > 1e-5)
     error('log_MH_r_Gamma', 'log MH r Gamma values do not match');
   end  
end

