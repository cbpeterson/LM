% For running locally
cd 'C:\Users\Marina\Desktop\LM only code\Code'
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\BVSSURV\'

% In case I need to debug
dbstop if error;

% Number of transcription factors
num_tfs = 40;

% Number of genes regulated by each transcription factor
g_per_tf = 5;

% Number of genes with sign changed in models 2 and 4
g_change = 2;

input_folder = './Simulation/Inputs/ReducedDim/';

% Sample size for training and test sets
n = 150;

% Total number of genes
p = num_tfs * (g_per_tf + 1);

% Since Z's are set to 0 here, this param does not matter
h_alpha = 0;

% Prior param for G-Wishart
delta_prior = 3;

% Number of MCMC iterations for Bayesian methods
burnin  = 10000;
nmc = 10000;

% Beta values as given in Li and Li for Model 1
beta_nonzero = cat(1, 5, repmat(5 / sqrt(10), g_per_tf, 1), ...
    -5, repmat(-5 / sqrt(10), g_per_tf, 1), ...
    3, repmat(3 / sqrt(10), g_per_tf, 1), ...
    -3, repmat(-3 / sqrt(10), g_per_tf, 1));

    
% Number of true predictors is number of nonzero beta
p_true = size(beta_nonzero, 1);
beta = cat(1, beta_nonzero, zeros(p - p_true, 1));

% Set up Sigma matrix
Sigma = eye(p);
tf_indices = 1:(g_per_tf + 1):p;

for tf = 1:num_tfs
    for gene = 1:g_per_tf
        % Correlation of transcription factor to genes it controls is 0.7
        Sigma(tf_indices(tf), tf_indices(tf) + gene) = 0.7;
        Sigma(tf_indices(tf) + gene, tf_indices(tf)) = 0.7;
        
        % Correlation of genes to each other will be 0.7^2
        for gene2 = (gene + 1):g_per_tf
            Sigma(tf_indices(tf) + gene, tf_indices(tf) + gene2) = 0.7^2;
            Sigma(tf_indices(tf) + gene2, tf_indices(tf) + gene) = 0.7^2;
        end
    end
end

% Check SNR
sigma_sq_e = sum(beta .^ 2) / 4;
SNR = beta' * Sigma * beta / sigma_sq_e;

% Variance tau^2
tau_sq = sigma_sq_e;

% Scale parameter for prior variance on nonzero betas
h_beta = var(beta_nonzero) / sigma_sq_e;

% Prior parameters

% Shape and scale of inverse gamma prior on tau^2
a_0 = 2;
b_0 = tau_sq;
    
% True graph structure
Omega_true = inv(Sigma);
Adj_true = abs(Omega_true) > 0.001;
    
% Vectorize unique entries of adjacency for assessing edge selection
% and only consider edges for true variables
indmx = reshape([1:p^2], p, p);
upperind = indmx(triu(indmx, 1) > 0);
Adj_true = Adj_true(upperind);
Adj_true = Adj_true(1:(p_true * (p_true - 1) / 2));

% True model: first p_true genes contribute to outcome
gamma_true = cat(1, ones(p_true, 1), zeros(p - p_true, 1));

% Prior param for G-Wishart
D_prior = eye(p);

% Parameters of MRF prior - how to determine proper settings for a and b?
% Li and Zhang discuss this, esp phase transition property
a = -9;
b = 0.05;
lambda_mrf = 0.01;

% Generate X
X = zeros(n, p);
for i = 1:n
    X(i, :) = rMNorm(zeros(p, 1), Sigma, 1)';
end

% Standardize X
X = zscore(X);

% Generate response variable Y with variance tau_sq
Y = zeros(n, 1);
for i = 1:n
    Y(i) = normrnd(X(i, :) * beta, sqrt(tau_sq));
end

% DON'T Center Y. Instead add 100
Y = Y + 100;

% Parameter h_0 affects variance of normal prior on intercept
h_0 = 1e6;

% True event times
T = exp(Y);

% Censor 35 events at random. dcen = censoring indicators
n_cens = 35;
dcen = ones(n, 1);
dcen(randsample(n, n_cens)) = 0;

% Substitute smaller time for these subjects
T_star = T;
T_star(dcen == 0) = T(dcen == 0) .* unifrnd(0.25, 0.75, n_cens, 1);

% Initial value of Omega (precision matrix) and gamma
Omega_init = eye(p);
gamma_init = zeros(p, 1);

% Run MCMC sampler for joint graph and variable selection
% Clinical covariates Z are set to all zeros here
% Since p is large, param summary_only is set to true
tic
[gamma_save, Omega_save, adj_save, ar_gamma, info, W_save] = MCMC_LM_GWishart_AFT(X, ...
    T_star, dcen, zeros(n, 5), ...
    a_0, b_0, h_0, h_alpha, h_beta, a, b, lambda_mrf, delta_prior, D_prior, ...
    gamma_init, Omega_init, burnin, nmc, true);
toc

% Get performance of graph structure learning

% Save plot of MCMC performance
% NOTE: may need to turn this off when running in batch mode on
% the cluster
h = plot(1:(burnin + nmc), sum(info.full_gamma, 1))
hold on
line([1,(burnin + nmc)], [p_true, p_true], 'Color', 'green')
plot(1:(burnin + nmc), sum(info.full_gamma(1:p_true, :), 1), 'Color', 'red')
hold off
xlabel('Iteration')
ylabel('Number of variables')
title('Green = true, red = true selections, blue = total selections')

ppi_var = mean(gamma_save, 2);
ppi_edges = adj_save;

% Get true and false positive rates for variable selection
sel_var = ppi_var > 0.5;
[tpr_var, fpr_var, mcc_var] = tpr_fpr_var(sel_var, gamma_true);

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

% Question: how similar are W estimates to true log(T) values for censored
% observations?
log_T_cen = (log(T));
log_T_cen = log_T_cen(dcen == 0, 1);
W_cen = mean(W_save, 2);
W_cen = W_cen(dcen == 0, 1);
scatter(log_T_cen, W_cen);
refline(1, 0);
xlabel('log(T)');
ylabel('MCMC estimate of W');
title(sprintf('Comparison between true and estimated survival\n times for censored observations'));
% This looks fine. Mostly want to see that all points line above the line
% but are reasonably close.

% In future, look at full MCMC results, see if these (or their running mean)
% approach true value

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
