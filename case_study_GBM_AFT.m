% Test performance using X matrix from case study
cd 'C:\Users\Marina\Desktop\LM only code\Code\branch3 - sample full graph';
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\glmnet_matlab';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\BVSSURV\'

% Load data from file
load '..\Case studies\TCPA - GBM\GBM_survival.mat';
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

% Initial value for gamma
gamma_init = zeros(p, 1);

% Number of MCMC iterations
burnin  = 2000;
nmc = 2000;

% Paramters to be tuned
a = -3;
b = 0.5;

% Parameter h_0 affects variance of normal prior on intercept
h_0 = 1e6;

% Note that the parameterization used in the code is slightly different from those in Wang (2014).
% (h in code) =  (h in paper )^2
h = 250^2;

% (v0 in code) = (v0 in paper)^2
v0 = 0.25^2;

% (v1 in code) = (v1 in paper)^2
v1 = h * v0;

lambda = 1;
pii = 2 / (p - 1);

rng(1435);

% Run MCMC sampler for joint graph and variable selection
tic
[gamma_save, Omega_save, adj_save, ar_gamma, info, W_save] = MCMC_LM_scalable_AFT(X, ...
    t_star, delta, Z, ...
    a_0, b_0, h_0, h_alpha, h_beta, a, b, v0, v1, lambda, pii, ...
    gamma_init, burnin, nmc, true);
toc

% Compare W estimates to censored times. Note that W_save is really log of
% times.
W_est = mean(W_save, 2);
obs_time = log(t_star);
scatter(log(t_star), mean(W_save, 2))
hold on;
scatter(obs_time(find(delta)), W_est(find(delta)), 6, 'red');
hold off;

% Save plot of MCMC performance
% NOTE: may need to turn this off when running in batch mode on
% the cluster
plot(1:(burnin + nmc), sum(info.full_gamma, 1))
xlabel('Iteration')
ylabel('Number of variables')
title('Blue = total selections')

ppi_var = mean(gamma_save, 2);
ppi_edges = adj_save;

% Get true and false positive rates for variable selection
sel_var = ppi_var > 0.5;

% Edges selected using marginal PPI threshold of 0.5
sel_edges = ppi_edges > 0.5;

% Which selected variables are connected?
sel_edges(sel_var, sel_var)

% How many edges are selected and what is overall sparsity level?
(sum(sum(sel_edges)) - p) / 2
(sum(sum(sel_edges)) - p) / p / (p-1)

S = X' * X / n;
Rho = my_parcorr(S);

