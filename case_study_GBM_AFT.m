% Test performance using X matrix from case study
cd '/hsgs/nobackup/cbp/LM_plus_graph/branch3';
addpath '/hsgs/nobackup/cbp/LM_plus_graph/glmnet_matlab';
addpath '/hsgs/nobackup/cbp/LM_plus_graph/BVSSURV';

% Load data from file
load '../Case_studies/TCPA_GBM/GBM_survival.mat';
[n, p] = size(X);

% Set seed to ensure that we end up with same training and tests sets and
% that results are reproducible
rng(9025);

% Divide into training and test sets at random
n_train = 175;
ind_train = randsample(n, n_train);
X_train = X(ind_train, :);
age_train = age(ind_train, 1);
gender_train = gender(ind_train, 1);
t_star_train = t_star(ind_train, 1);
delta_train = delta(ind_train, 1);

% Standardize predictors
X_train = zscore(X_train);

% Covariates Z
Z_train = [age_train, gender_train];

% Standardize these
Z_train = zscore(Z_train);

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
burnin  = 10000;
nmc = 10000;

% Paramters to be tuned
a = -1.5;
fprintf('a = %g\n', a);
b = 0.5;
fprintf('b = %g\n', b);

% Parameter h_0 affects variance of normal prior on intercept
h_0 = 1e6;

% Note that the parameterization used in the code is slightly different from those in Wang (2014).
% (h in code) =  (h in paper )^2
h = 600^2;

% (v0 in code) = (v0 in paper)^2
v0 = 0.6^2;

% (v1 in code) = (v1 in paper)^2
v1 = h * v0;

lambda = 1;
pii = 2 / (p - 1);

% Run MCMC sampler for joint graph and variable selection
tic
[gamma_save, Omega_save, adj_save, ar_gamma, info, W_save] = MCMC_LM_scalable_AFT(X_train, ...
    t_star_train, delta_train, Z_train, ...
    a_0, b_0, h_0, h_alpha, h_beta, a, b, v0, v1, lambda, pii, ...
    gamma_init, burnin, nmc, true);
toc

% Compare W estimates to censored times. Note that W_save is really log of
% times.
W_est = mean(W_save, 2);
obs_time = log(t_star_train);
scatter(log(t_star_train), mean(W_save, 2))
hold on;
scatter(obs_time(find(delta_train)), W_est(find(delta_train)), 6, 'red');
hold off;

% Save plot of MCMC performance
% NOTE: may need to turn this off when running in batch mode on
% the cluster
trace_plot = plot(1:(burnin + nmc), sum(info.full_gamma, 1))
xlabel('Iteration')
ylabel('Number of variables')
title('Blue = total selections')
saveas(trace_plot, strcat('./Output/MCMC_traceplot_GBM_ntrain', num2str(n_train), '_a', ...
    num2str(a), '_b', num2str(b), '.png'), 'png');

ppi_var = mean(gamma_save, 2);
ppi_edges = adj_save;

% Get true and false positive rates for variable selection
sel_var = ppi_var > 0.5;
fprintf('Number of selected variables = %g\n', sum(sel_var));
fprintf('Variable selections:');
find(sel_var)

% Edges selected using marginal PPI threshold of 0.5
sel_edges = ppi_edges > 0.5;

% Which selected variables are connected?
sel_edges(sel_var, sel_var)

% How many edges are selected and what is overall sparsity level?
(sum(sum(sel_edges)) - p) / 2
(sum(sum(sel_edges)) - p) / p / (p-1)
