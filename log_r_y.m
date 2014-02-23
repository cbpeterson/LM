function [log_y_mh_ratio] = log_r_y(gamma, gamma_prop, X, Y, Z, h_alpha, h_beta, a_0, b_0)
% Compute MH ratio p(Y|gamma_prop) / p(Y|gamma) on log scale

[n, p] = size(X);
X_gamma = X(:, find(gamma));
X_gamma_prop = X(:, find(gamma_prop));

% Use logdet rather than log(det()) is case det is large/small
% Similarly, log1p computes log(1 + p) which is accurate for small p
log_y_mh_ratio = 0.5 * logdet(eye(n) + h_alpha * Z * Z' + h_beta * X_gamma * X_gamma', 'chol') + ...
    (n + 2 * a_0) / 2 * log1p(1 / 2 / b_0 * Y' * ...
    inv(eye(n) + h_alpha * Z * Z' + h_beta * X_gamma * X_gamma') * Y) - ...
    0.5 * logdet(eye(n) + h_alpha * Z * Z' + h_beta * X_gamma_prop * X_gamma_prop', 'chol') - ...
    (n + 2 * a_0) / 2 * log1p(1 / 2 / b_0 * Y' * ...
    inv(eye(n) + h_alpha * Z * Z' + h_beta * X_gamma_prop * X_gamma_prop') * Y);

%{
  % Test: compare to ratio of multivariate t distributions with given params
  df = 2 * a_0;
  scale = b_0 / a_0 * (eye(n) + h_alpha * Z * Z' + h_beta * X_gamma * X_gamma');
  scale_prop =  b_0 / a_0 * (eye(n) + h_alpha * Z * Z' + h_beta * X_gamma_prop * X_gamma_prop');
  
  comp = mvtLogpdf(Y', 0, scale_prop, df) -  mvtLogpdf(Y', 0, scale, df);
  % Confirmed 7/16/2013 that comp = ratio computed above
    %}
end