% For running locally
cd 'C:\Users\Marina\Desktop\LM only code\Code\branch2 - independent noise'
addpath '.\rmnvrnd';
addpath '.\logmvnpdf';
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\Case studies\Leukemia';

% In case I need to debug
dbstop if error;

% Read in training response variables W
W = dlmread('leukY.txt');

% Read in predictor matrix for training data
X = dlmread('leukX.txt');

% Standardize X
X = zscore(X);

% Sample covariance
S = cov(X);
% Partial correlation can't be estimated since n < p

% Sample size & number of variables
[n p] = size(X);

% Since we are not currently included covariates, h_alpha does not matter
% Set h_beta to 1 since predictors are standardized 
h_alpha = 0;
h_beta = 1;

% Prior params for G-Wishart
delta_prior = 3;
D_prior = eye(p);

% Number of MCMC iterations
burnin  = 10000;
nmc = 10000;

% Parameters of MRF prior - how to determine proper settings for a and b?
% Li and Zhang discuss this, esp phase transition property
a = -5.2;
b = 0.01;
lambda_mrf = 0.001;

% Parameter h_0 affects variance of normal prior on intercept
h_0 = 1e6;

% Initial value of Omega (precision matrix) and gamma
Omega_init = eye(p);
gamma_init = zeros(p, 1);

% Check results from lasso
[B_lasso, FitInfo] = lassoglm(X, W, 'binomial', 'Link', 'probit');
% Can check against these results to see if top vars are similar

% Top vars from lasso
find(B_lasso(:, 62))
% 12, 16, 32, 33, 99, 105

% Run MCMC sampler for joint graph and variable selection
% Clinical covariates Z are set to all zeros here
% Since p is large, param summary_only is set to true
tic
[gamma_save, Omega_save, adj_save, ar_gamma, info, Y_save] = MCMC_LM_GWishart_probit(X, ...
    W, zeros(n, 5), ...
    h_0, h_alpha, h_beta, a, b, lambda_mrf, delta_prior, D_prior, ...
    gamma_init, Omega_init, burnin, nmc, true);
toc

% Get performance of graph structure learning

% Save plot of MCMC performance
% NOTE: may need to turn this off when running in batch mode on
% the cluster
h = plot(1:(burnin + nmc), sum(info.full_gamma, 1))
xlabel('Iteration')
ylabel('Number of variables')
title('Blue = total selections')

ppi_var = mean(gamma_save, 2);
ppi_edges = adj_save;

% Get true and false positive rates for variable selection
sel_var = ppi_var > 0.5;

% Edges selected using marginal PPI threshold of 0.5
sel_edges = ppi_edges > 0.5;

% Check selected edges against true graph (i.e. true edges
% among true variables)
sel_edges = sel_edges(upperind);
sel_edges = sel_edges(1: (p_true * (p_true - 1) / 2));
tp_edges = sum(sel_edges & Adj_true);
fp_edges = sum(sel_edges & ~Adj_true);
tn_edges = sum(~sel_edges & ~Adj_true);
fn_edges = sum(~sel_edges & Adj_true);

% True positive rate and false positive rate for edge
% selection among true variables
tpr_edges = tp_edges / (tp_edges + fn_edges);
fpr_edges = fp_edges / (fp_edges + tn_edges);

fprintf('edge_tpr = %g\n', tpr_edges);
fprintf('edge_fpr = %g\n', fpr_edges);

% Question: how similar are Y estimates to true Y values?
Y_est = mean(Y_save, 2);
scatter(Y_true, Y_est, 10);
hold on
scatter(Y_true(W == 1), Y_est(W == 1), 10, 'red');
hold off
xlabel('True Y');
ylabel('MCMC estimate of Y');
title(sprintf('Comparison between true and estimated Y values'));

% Check if we approach correct W value across iterations for censored W
% Choose a censored value at random
cens_ind = find(~dcen, 1);
h = plot(1:(burnin + nmc), info.full_W(cens_ind, :))
hold on
line([1,(burnin + nmc)], [log(T(cens_ind)), log(T(cens_ind))], 'Color', 'green')
line([1,(burnin + nmc)], [log(T_star(cens_ind)), log(T_star(cens_ind))], 'Color', 'red')
% MCMC average
% line([1,(burnin + nmc)], [mean(W_save(cens_ind, :)), mean(W_save(cens_ind, :))], 'Color', 'yellow')
hold off
xlabel('Iteration')
ylabel('Value of W')
title('Green = true value of log(T), red = log(T star), blue = sampled W values')
