% For running locally
cd 'C:\Users\Marina\Desktop\LM only code\Code\branch3 - sample full graph'
addpath '.\rmnvrnd';
addpath '.\logmvnpdf';
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\Case studies\Imaging';

% In case I need to debug
dbstop if error;

% Read in training response variables W
W = dlmread('Y.txt');

% Read in predictor matrix for training data
X = dlmread('X.txt', '\t', 1, 0);

% Standardize X
X = zscore(X);

% Sample covariance
S = cov(X);

% Partial correlation matrix
Rho = my_parcorr(S);

% Sample size & number of variables
[n p] = size(X);

% Since we are not currently included covariates, h_alpha does not matter
% Set h_beta to 1 since predictors are standardized 
h_alpha = 0;
h_beta = 1;

% Number of MCMC iterations
burnin  = 2000;
nmc = 2000;

% Use parameters from Stingo imaging paper
a = -2.5;
b = 0.5;

% Parameter h_0 affects variance of normal prior on intercept
h_0 = 1e6;

% Initial value of gamma
gamma_init = zeros(p, 1);

% Check results from glm fit

% Results are junk - due to "perfect separation"
[beta_glm, beta_dev, glm_stats] = glmfit(X, W, 'binomial', 'probit'); 
scatter([0; beta], beta_glm)
[[0; beta], beta_glm]

% Does regulatization help? YES, lasso is able to handle this
[B_lasso, FitInfo] = lassoglm(X, W, 'binomial', 'Link', 'probit');
% Can check against these results to see if top vars are similar
find(B_lasso(:, 88))
% Some good vars as identified by lasso

% Note that the parameterization used in the code is slightly different from those in Wang (2014).
% (h in code) =  (h in paper )^2
h = 300^2;

% (v0 in code) = (v0 in paper)^2
v0 = 0.35^2;

% (v1 in code) = (v1 in paper)^2
v1 = h * v0;

lambda = 1;
pii = 2 / (p - 1);

rng(34689);

% Run MCMC sampler for joint graph and variable selection
% Clinical covariates Z are set to all zeros here
% Since p is large, param summary_only is set to true
tic
[gamma_save, Omega_save, adj_save, ar_gamma, info, Y_save] = MCMC_LM_scalable_probit(X, ...
    W, zeros(n, 5), ...
    h_0, h_alpha, h_beta, a, b, v0, v1, lambda, pii, ...
    gamma_init, burnin, nmc, true);
toc

% Get performance of graph structure learning

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

% How many edges are selected and what is overall sparsity level?
(sum(sum(sel_edges)) - p) / 2
(sum(sum(sel_edges)) - p) / p / (p-1)


