% Test performance using X matrix from case study
cd 'C:\Users\Marina\Desktop\LM only code\Code\';
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\glmnet_matlab';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\BVSSURV\'

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
burnin  = 500;
nmc = 500;

% Paramters to be tuned
a = -5;
b = 0.05;
lambda_mrf = 0.01;

% Run MCMC sampler for joint graph and variable selection
tic
[gamma_save, Omega_save, adj_save, ar_gamma, info, W_save] = MCMC_LM_GWishart_AFT(X, ...
    t_star, delta, Z, ...
    a_0, b_0, h_alpha, h_beta, a, b, lambda_mrf, delta_prior, D_prior, ...
    gamma_init, Omega_init, burnin, nmc, true);
toc