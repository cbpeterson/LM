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
burnin  = 10000;
nmc = 10000;

% Use parameters from Stingo imaging paper
a = -4.5;
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
h = 100^2;

% (v0 in code) = (v0 in paper)^2
v0 = 0.1^2;

% (v1 in code) = (v1 in paper)^2
v1 = h * v0;

lambda = 1;
pii = 2 / (p - 1);

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
