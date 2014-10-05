function [log_p] = log_X_not_included(X_not_incl, X_incl, h_gamma, k_0, delta_c)
  p_gamma = size(X_incl, 2);
  p_not_gamma = size(X_not_incl, 2);
  n = size(X_incl, 1);
  U_inv = X_incl' * X_incl + 1 / h_gamma * eye(p_gamma);
  Psi = k_0 * eye(p_not_gamma) + ...
        X_not_incl' / (eye(n) + h_gamma * (X_incl * X_incl')) * X_not_incl;
  
  log_p = logMvGamma((delta_c + n) / 2, p_not_gamma) + ...
      delta_c / 2 * log(k_0) - ...
      logMvGamma(delta_c / 2, p_not_gamma) - n * p_not_gamma / 2 * log(pi) - ...
      p_not_gamma / 2 * log(h_gamma) - ...
      p_not_gamma / 2 * logdet(U_inv) - ...
      (delta_c + n) / 2 * logdet(Psi);
end