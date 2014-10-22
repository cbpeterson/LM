function [log_y_mh_ratio] = log_r_y_probit(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta)
% Compute MH ratio p(Y|gamma_prop) / p(Y|gamma) on log scale

[n, p] = size(X);
X_gamma = X(:, find(gamma));
X_gamma_prop = X(:, find(gamma_prop));

Sigma = eye(n) + h_0 * (ones(n, 1) * ones(1, n)) + h_alpha * (Z * Z') + h_beta * (X_gamma * X_gamma');
Sigma_prop = eye(n) + h_0 * (ones(n, 1) * ones(1, n)) + h_alpha * (Z * Z') + h_beta * (X_gamma_prop * X_gamma_prop');
mu = zeros(1, n); 

log_y_mh_ratio = logmvnpdf(Y', mu, Sigma_prop) - logmvnpdf(Y', mu, Sigma);
end