% Test performance using X matrix from case study with standard linear
% model used to simulate response
cd 'C:\Users\Marina\Desktop\LM only code\Code\';
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\glmnet_matlab';

% Load data from file
load 'Case study\GBM_survival';
[n, p] = size(X);

% Standardize predictors
X = zscore(X);

% Empirical covariance matrix and precision matrix
Sigma = X' * X / n;
Omega = inv(Sigma);

% Covariates Z
Z = [age, gender];

% Standardize these
Z = zscore(Z);

% Shape and scale of inverse gamma prior on tau^2
% Parameters chosen based on recmmendations in Stingo paper
a_0 = 3;
b_0 = 0.5;

% Since covariates and predictors are standardzed, set these to 1
% Also based on recommendation in Stingo paper to set h based on variance of
% covariates
h_alpha = 1;
h_beta = 1;

% Prior params for G-Wishart
delta_prior = 3;
D_prior = eye(p);

% Initial values for gamma and Omega
gamma_init = zeros(p, 1);
Omega_init = eye(p);

% Number of MCMC iterations for Bayesian methods
burnin  = 10000;
nmc = 10000;

% Paramters to be tuned
a = -11;
b = 0.001;
lambda_mrf = 0.001;

% Compute partial correlations given Omega
rho = zeros(p, p);
for i = 1:p
    for j = 1:p
        if i == j
            rho(i, j) = 1;
        else
            rho(i, j) = -Omega(i, j) ./ sqrt(Omega(i, i) .* ...
                Omega(j, j));
        end
    end
end

% Threshold rho based on abs value to get adjacency matrix
adj = rho;
adj(abs(adj) < 0.5) = 0;
adj(abs(adj) >= 0.5) = 1;
adj = adj - eye(p);
bg = biograph(adj);
view(bg);

% Indices corresponding to reasonably disconnect modules
gamma_true_inds = [52, 53, 144, 65, 160, 62, ...
    78, 177, 116, ...
    103, 172, 39, 43, 182, 72, 51, 163, 4];
gamma_true = zeros(p, 1);
gamma_true(gamma_true_inds) = 1;

% Number of true predictors
p_true = sum(gamma_true);

% Coefficients
beta = gamma_true * 5;

% Check SNR
sigma_sq_e = sum(beta .^ 2) / 4;
SNR = beta' * Sigma * beta / sigma_sq_e;

% Variance tau^2
tau_sq = sigma_sq_e;

% True graph structure
Omega_true = inv(Sigma);
Adj_true = abs(Omega_true) > 0.001;

% Generate response variable Y with variance tau_sq
Y = zeros(n, 1);
for i = 1:n
    Y(i) = normrnd(X(i, :) * beta, sqrt(tau_sq));
end

% Center Y
Y = Y - repmat(mean(Y, 1), n, 1);

% Run MCMC sampler
% Clinical covariates Z are set to all zeros here
[gamma_save, Omega_save, adj_save, ar_gamma, info] = MCMC_LM_GWishart_simplified(X, Y, zeros(n, 5), ...
    a_0, b_0, h_alpha, h_beta, a, b, lambda_mrf, delta_prior, D_prior, ...
    gamma_init, Omega_init, burnin, nmc, true);

% Generate plot of MCMC performance
h = plot(1:(burnin + nmc), sum(info.full_gamma, 1))
hold on
line([1,(burnin + nmc)], [p_true, p_true], 'Color', 'green')
plot(1:(burnin + nmc), sum(info.full_gamma(gamma_true_inds, :), 1), 'Color', 'red')
hold off
xlabel('Iteration')
ylabel('Number of variables')
title('Green = true, red = true selections, blue = total selections')

% Do we select modules correctly?
ind_module1 = [52, 53, 144, 65, 160, 62];
ind_module2  = [78, 177, 116];
ind_module3 = [103, 172, 39, 43, 182, 72, 51, 163, 4];

sum(info.full_gamma(ind_module1, burnin + nmc))
sum(info.full_gamma(ind_module2, burnin + nmc))
sum(info.full_gamma(ind_module3, burnin + nmc))

% Check: do selected variables have large partial correlations?
adj = rho;
adj(abs(adj) < 0.4) = 0;
adj(abs(adj) >= 0.4) = 1;
adj = adj - eye(p);
ind_sel = find(info.full_gamma(:, burnin + nmc));
bg = biograph(adj(ind_sel, ind_sel));
view(bg);

% Yes, it does seem that many are correlated.
% Maybe try increasing signal strength?



