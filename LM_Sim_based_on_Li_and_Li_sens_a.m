% For running on the cluster
% Assume argument "run" is defined on the command line and is between 1 and 200
addpath '/hsgs/nobackup/cbp/LM_plus_graph/branch3/';
addpath '/hsgs/nobackup/cbp/LM_plus_graph/glmnet_matlab';

niter = 50;
cur_iter = mod(run, niter);
if cur_iter == 0
    cur_iter = niter;
end

% Assume we are always in model 1, but parameters are changing
model = 1;
a_setting = (run - cur_iter) / 50 + 1;
cd(fullfile(strcat('/hsgs/nobackup/cbp/LM_plus_graph/branch3/Output/Simulation/Sensitivity/a_setting', num2str(a_setting))))

% Number of transcription factors
num_tfs = 40;

% Number of genes regulated by each transcription factor
g_per_tf = 5;

% Number of genes with sign changed in models 2 and 4
g_change = 2;

input_folder = '/hsgs/nobackup/cbp/LM_plus_graph/Simulation/Inputs/ReducedDim/';

% Sample size for training and test sets
n = 100;

% Total number of genes
p = num_tfs * (g_per_tf + 1);

% Since Z's are set to 0 here, this param does not matter
h_alpha = 0;

% Number of metrics and methods to compare for this model/iteration
nmetric = 4;
nmethod = 5;

% Number of MCMC iterations for Bayesian methods
burnin  = 5000;
nmc = 5000;
    
beta_nonzero = 0;
switch model
    case 1
        % Beta values as given in Li and Li for Model 1
        beta_nonzero = cat(1, 5, repmat(5 / sqrt(10), g_per_tf, 1), ...
            -5, repmat(-5 / sqrt(10), g_per_tf, 1), ...
            3, repmat(3 / sqrt(10), g_per_tf, 1), ...
            -3, repmat(-3 / sqrt(10), g_per_tf, 1));
    case 2
        % Beta values as given in Li and Li for Model 2
        beta_nonzero = cat(1, 5, ...
            repmat(-5 / sqrt(10), g_change, 1), ...
            repmat(5 / sqrt(10), g_per_tf - g_change, 1), ...
            -5, repmat(5 / sqrt(10), g_change, 1), ...
            repmat(-5 / sqrt(10), g_per_tf - g_change, 1), ...
            3, repmat(-3 / sqrt(10), g_change, 1), ...
            repmat(3 / sqrt(10), g_per_tf - g_change, 1), ...
            -3, repmat(3 / sqrt(10), g_change, 1), ...
            repmat(-3 / sqrt(10), g_per_tf - g_change, 1));
    case 3
        % Beta values as given in Li and Li for Model 3
        beta_nonzero = cat(1, 5, repmat(5 / 10, g_per_tf, 1), ...
            -5, repmat(-5 / 10, g_per_tf, 1), ...
            3, repmat(3 / 10, g_per_tf, 1), ...
            -3, repmat(-3 / 10, g_per_tf, 1));
    case 4
        % Beta values as given in Li and Li for Model 4
        beta_nonzero = cat(1, 5, ...
            repmat(-5 / 10, g_change, 1), ...
            repmat(5 / 10, g_per_tf - g_change, 1), ...
            -5, repmat(5 / 10, g_change, 1), ...
            repmat(-5 / 10, g_per_tf - g_change, 1), ...
            3, repmat(-3 / 10, g_change, 1), ...
            repmat(3 / 10, g_per_tf - g_change, 1), ...
            -3, repmat(3 / 10, g_change, 1), ...
            repmat(-3 / 10, g_per_tf - g_change, 1));
end

switch a_setting
    case 1
      a = -3.5;
    case 2
      a = -3.25;
    case 3
      a = -3;
    case 4
      a = -2.75;
    case 5
      a = -2.5;
    case 6
      a = -2.25;
end


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

% Calculate graph Laplacian and associated values
L = zeros(p, p);

% Use edge indicator as weight matrix
W = Adj_true - eye(p);

% Degree of each node
d = sum(W, 1);

% L computed as on p. 1176 in Li and Li
for u = 1:p
    for v = 1:p
        if u == v && d(u) ~= 0
            L(u, v) = 1 - W(u, v) / d(u);
        elseif W(u, v) > 0
            L(u, v) = -W(u, v) / sqrt(d(u) * d(v));
        end
    end
end

% Calulcate S matrix
[U, T] = schur(L, 'real');
S = U * sqrt(T);

% Vectorize unique entries of adjacency for assessing edge selection
% and only consider edges for true variables
indmx = reshape([1:p^2], p, p);
upperind = indmx(triu(indmx, 1) > 0);
Adj_true = Adj_true(upperind);

% True model: first p_true genes contribute to outcome
gamma_true = cat(1, ones(p_true, 1), zeros(p - p_true, 1));

% Parameters of MRF prior - how to determine proper settings for a and b?
% Li and Zhang discuss this, esp phase transition property
b = 0.5;

% Prior probability of variable inclusion for Bayesian variable selection
lambda_bvs = p_true / p;

% Set random number seed to be specific to iter and model for
% reproducibility
rng(model * 1717 + cur_iter * 3);

% Record performance summmary for current model
% Dim 1: 4 = sens, spec, mcc, pmse
% Dim 2: 5 = lasso, en, LL, selection using graph, selection w/o graph
cur_perf_summary = zeros(nmetric, nmethod);

fprintf('cur iteration = %d\n', cur_iter);

% Read in simulation data from file
X = csvread(strcat(input_folder, 'X_', num2str(cur_iter), '_model', num2str(model), '.csv'));
X_test = csvread(strcat(input_folder, 'X_test_', num2str(cur_iter), '_model', num2str(model), '.csv'));
Y = csvread(strcat(input_folder, 'Y_', num2str(cur_iter), '_model', num2str(model), '.csv'));
Y_test = csvread(strcat(input_folder, 'Y_test_', num2str(cur_iter), '_model', num2str(model), '.csv'));

% LASSO --------------------------------------------------------------
% Fit lasso with 10-fold CV
% Default setting for alpha is 1 (i.e. lasso)
lasso_opts = glmnetSet;
lasso_cv = cvglmnet(X, Y, 10, [], 'response', 'gaussian', lasso_opts, 0);
best_lambda = lasso_cv.lambda_min;
lasso_fit = lasso_cv.glmnet_object;
best_index = find(lasso_fit.lambda == best_lambda);
best_beta_lasso = lasso_fit.beta(:, best_index);

% Get tpr and fpr of variable selection
sel_lasso = best_beta_lasso ~= 0;
[tpr_lasso, fpr_lasso, mcc_lasso] = tpr_fpr_var(sel_lasso, gamma_true);

% Now do prediction
lasso_pred = X_test * best_beta_lasso;

% Calculate pmse
pmse_lasso = mean((lasso_pred - Y_test) .^ 2);

% Store results
cur_perf_summary(1, 1) = tpr_lasso;
cur_perf_summary(2, 1) = 1 - fpr_lasso;
cur_perf_summary(3, 1) = mcc_lasso;
cur_perf_summary(4, 1) = pmse_lasso;

% ELASTIC NET --------------------------------------------------------
% Fit elastic net with 10-fold CV

% Grid of alpha values to search
alpha_vals = 0:0.04:1;

% Default parameter settings
en_opts = glmnetSet;

% Search over alphas to find min cv error
nalpha = size(alpha_vals, 2);
lowest_cv_alpha = Inf;
best_beta_en = zeros(p, 1);
for cur_cv_iter = 1:nalpha
    cur_alpha = alpha_vals(1, cur_cv_iter);
    en_opts.alpha = cur_alpha;
    en_cv = cvglmnet(X, Y, 10, [], 'response', 'gaussian', en_opts, 0);
    min_cv_err = min(en_cv.cvm);
    if min_cv_err < lowest_cv_alpha
        lowest_cv_alpha = min_cv_err;
        best_alpha = cur_alpha;
        best_lambda = en_cv.lambda_min;
        en_fit = en_cv.glmnet_object;
        best_index = find(en_fit.lambda == best_lambda);
        best_beta_en = en_fit.beta(:, best_index);
    end
end

% Get tpr and fpr of variable selection
sel_en = best_beta_en ~= 0;
[tpr_en, fpr_en, mcc_en] = tpr_fpr_var(sel_en, gamma_true);

% Now do prediction
en_pred = X_test * best_beta_en;

% Calculate pmse
pmse_en = mean((en_pred - Y_test) .^ 2);

% Store results
cur_perf_summary(1, 2) = tpr_en;
cur_perf_summary(2, 2) = 1 - fpr_en;
cur_perf_summary(3, 2) = mcc_en;
cur_perf_summary(4, 2) = pmse_en;

% LI AND LI PROCEDURE -------------------------------------------------
% Augmented Y
Y_star = cat(1, Y, zeros(p, 1));

% Search over grid of lambda2 values
lambda2_exp = -3:0.2:2;
lambda2_vals = 10 .^ lambda2_exp;

% Search over lambda2s to find min cv error
nlambda2 = size(lambda2_vals, 2);
lowest_cv_lambda2 = Inf;
best_lambda2 = Inf;
best_beta_star_LL = zeros(p, 1);
for cur_cv_iter = 1:nlambda2
    lambda2 = lambda2_vals(1, cur_cv_iter);
    
    % Augmented X
    X_star = (1 + lambda2) ^ (-1/2) * cat(1, X, sqrt(lambda2) * S');
    
    % Solve lasso problem for augmented X and Y
    LL_cv = cvglmnet(X_star, Y_star, 10, [], 'response', 'gaussian', lasso_opts, 0);
    min_cv_err = min(LL_cv.cvm);
    if min_cv_err < lowest_cv_lambda2
        lowest_cv_lambda2 = min_cv_err;
        best_lambda2 = lambda2;
        best_lambda = LL_cv.lambda_min;
        LL_fit = LL_cv.glmnet_object;
        best_index = find(LL_fit.lambda == best_lambda);
        best_beta_star_LL = LL_fit.beta(:, best_index);
    end
end

% Get tpr and fpr of variable selection
sel_LL = best_beta_star_LL ~= 0;
[tpr_LL, fpr_LL, mcc_LL] = tpr_fpr_var(sel_LL, gamma_true);

% Now do prediction
beta_LL = 1 / sqrt(1 + best_lambda2) * best_beta_star_LL;
LL_pred = X_test * beta_LL;

% Calculate pmse
pmse_LL = mean((LL_pred - Y_test) .^ 2);

% Store results
cur_perf_summary(1, 3) = tpr_LL;
cur_perf_summary(2, 3) = 1 - fpr_LL;
cur_perf_summary(3, 3) = mcc_LL;
cur_perf_summary(4, 3) = pmse_LL;

% STANDARD BVS --------------------------------------------------------
% Initial value of gamma (variable selection indicators)
gamma_init = zeros(p, 1);

% Bayesian variable selection, not accounting for graph structure
[gamma_save] = MCMC_LM_no_graph_simplified(X, Y, zeros(n, 5), ...
    a_0, b_0, h_alpha, h_beta, lambda_bvs, gamma_init, burnin, nmc);
ppi_nograph = mean(gamma_save, 2);

% Check variable selections against true values
sel_nograph = ppi_nograph > 0.5;
[tpr_nograph, fpr_nograph, mcc_nograph] = tpr_fpr_var(sel_nograph, gamma_true);

% Compute prediction as MCMC avg prediction given var selection
BVS_pred = zeros(n, 1);
for i = 1:nmc
    p_gamma = sum(gamma_save(:, i));
    X_sel = X(:, logical(gamma_save(:, i)));
    X_test_sel = X_test(:, logical(gamma_save(:, i)));
    beta_hat = (X_sel' * X_sel + 1 / h_beta * eye(p_gamma)) \ X_sel' * Y;
    BVS_pred = BVS_pred + X_test_sel * beta_hat / nmc;
end

% Calculate pmse
pmse_BVS = mean((BVS_pred - Y_test) .^ 2);

% Store results
cur_perf_summary(1, 4) = tpr_nograph;
cur_perf_summary(2, 4) = 1 - fpr_nograph;
cur_perf_summary(3, 4) = mcc_nograph;
cur_perf_summary(4, 4) = pmse_BVS;

% PROPOSED METHOD -----------------------------------------------------
% Fix some hyperparameters
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
[gamma_save, Omega_save, adj_save, ar_gamma, info] = MCMC_LM_scalable_simplified(X, Y, zeros(n, 5), ...
    a_0, b_0, h_alpha, h_beta, a, b, v0, v1, lambda, pii, ...
    gamma_init, burnin, nmc, true);
toc

% Collect additional performance information for proposed
% method
csvwrite(strcat('./mh_info_model', num2str(model), '_iter', ...
    num2str(cur_iter), '.csv'), ...
    [info.n_add_prop', info.n_add_accept' ...
    info.n_remove_prop', info.n_remove_accept']);
csvwrite(strcat('./full_gamma__model', num2str(model), '_iter', ...
    num2str(cur_iter), '.csv'), info.full_gamma');
csvwrite(strcat('./node_degrees_model', num2str(model), '_iter', ...
    num2str(cur_iter), '.csv'), info.node_degrees');

% Get performance of graph structure learning

% Save plot of MCMC performance
% NOTE: may need to turn this off when running in batch mode on
% the cluster
h = plot(1:(burnin + nmc), sum(info.full_gamma, 1));
hold on;
line([1,(burnin + nmc)], [p_true, p_true], 'Color', 'green');
plot(1:(burnin + nmc), sum(info.full_gamma(1:p_true, :), 1), 'Color', 'red');
hold off;
xlabel('Iteration');
ylabel('Number of variables');
title('Green = true, red = true selections, blue = total selections');
saveas(h, strcat('./MCMC_traceplot_model', num2str(model), '_iter', ...
    num2str(cur_iter), '.png'), 'png');

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
tp_edges = sum(sel_edges & Adj_true);
fp_edges = sum(sel_edges & ~Adj_true);
tn_edges = sum(~sel_edges & ~Adj_true);
fn_edges = sum(~sel_edges & Adj_true);

% True positive rate and false positive rate for edge
% selection among true variables
[tpr_edges, fpr_edges, mcc_edges] = tpr_fpr_var(sel_edges, Adj_true);

fprintf('edge_tpr = %g\n', tpr_edges);
fprintf('edge_fpr = %g\n', fpr_edges);

% Compute prediction as MCMC avg prediction given var selection
my_pred = zeros(n, 1);
for i = 1:nmc
    p_gamma = sum(gamma_save(:, i));
    X_sel = X(:, logical(gamma_save(:, i)));
    X_test_sel = X_test(:, logical(gamma_save(:, i)));
    beta_hat = (X_sel' * X_sel + 1 / h_beta * eye(p_gamma)) \ X_sel' * Y;
    my_pred = my_pred + X_test_sel * beta_hat / nmc;
end

% Calculate pmse
my_pmse = mean((my_pred - Y_test) .^ 2);

% Store results
cur_perf_summary(1, 5) = tpr_var;
cur_perf_summary(2, 5) = 1 - fpr_var;
cur_perf_summary(3, 5) = mcc_var;
cur_perf_summary(4, 5) = my_pmse;

% Record performance for current iteration
csvwrite(strcat('./perf_summary_', num2str(model), '_iter', ...
    num2str(cur_iter), '.csv'), cur_perf_summary);
csvwrite(strcat('./perf_edges_', num2str(model), '_iter', ...
    num2str(cur_iter), '.csv'), [tpr_edges, fpr_edges, mcc_edges]);



