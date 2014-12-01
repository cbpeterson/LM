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
n_test = n - n_train;
ind_train = randsample(n, n_train);

% Set up training set
X_train = X(ind_train, :);
age_train = age(ind_train, 1);
gender_train = gender(ind_train, 1);
t_star_train = t_star(ind_train, 1);
delta_train = delta(ind_train, 1);

% Set up test set
ind_test = ones(n, 1);
ind_test(ind_train) = 0;
ind_test = find(ind_test);
X_test = X(ind_test, :);
age_test = age(ind_test, 1);
gender_test = gender(ind_test, 1);
t_star_test = t_star(ind_test, 1);
delta_test = delta(ind_test, 1);

% Standardize predictors
X_train = zscore(X_train);
X_test = zscore(X_test);

% Covariates Z
Z_train = [age_train, gender_train];
Z_test = [age_test, gender_test];

% Standardize these
Z_train = zscore(Z_train);
Z_test = zscore(Z_test);

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
a = -1.75;
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

csvwrite(strcat('./Output/mh_info_GBM_a_', num2str(a), '_b_', ...
    num2str(b), '.csv'), ...
    [info.n_add_prop', info.n_add_accept' ...
    info.n_remove_prop', info.n_remove_accept']);
csvwrite(strcat('./Output/full_gamma_GBM_a', num2str(a), '_b_', ...
    num2str(b), '.csv'), info.full_gamma');
csvwrite(strcat('./Output/gamma_save_GBM_a', num2str(a), '_b_', ...
    num2str(b), '.csv'), gamma_save);
csvwrite(strcat('./Output/W_save_GBM_a', num2str(a), '_b_', ...
    num2str(b), '.csv'), W_save);   

% Compare W estimates to censored times. Note that W_save is really log of
% times.
W_est = mean(W_save, 2);
obs_time = log(t_star_train);
scatter(log(t_star_train), mean(W_save, 2))
hold on;
scatter(obs_time(find(delta_train)), W_est(find(delta_train)), 6, 'red');
hold off;

% Save plot of MCMC performance
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

% Write ppi matrix and edge selections to disk
csvwrite(strcat('./Output/ppi_edges_GBM_a_', num2str(a), '_b_', ...
    num2str(b), '.csv'), ppi_edges);
csvwrite(strcat('./Output/sel_edges_GBM_a_', num2str(a), '_b_', ...
    num2str(b), '.csv'), sel_edges);

% Which selected variables are connected?
sel_edges(sel_var, sel_var)

% How many edges are selected and what is overall sparsity level?
(sum(sum(sel_edges)) - p) / 2
(sum(sum(sel_edges)) - p) / p / (p-1)

% Compute prediction as MCMC avg prediction given var selection
% Also compute estimated survival curve to be used with Brier score
upper_time = max(log(t_star_test(delta_test == 0)));
ycutsBS = 0:0.1:upper_time;
cutnum = length(ycutsBS);
my_pred = zeros(n_test, 1);
SurvFunc = zeros(n_test, cutnum);
for i = 1:nmc
    p_gamma = sum(gamma_save(:, i));
    X_train_sel = X_train(:, logical(gamma_save(:, i)));
    X_test_sel = X_test(:, logical(gamma_save(:, i)));
    
    % Need to include fixed covariates + intercept term
    all_cov_train = [X_train_sel, Z_train, ones(n_train, 1)];
    all_cov_test = [X_test_sel, Z_test, ones(n_test, 1)];
    coef_hat = (all_cov_train' * all_cov_train + diag([repmat(1 / h_beta, p_gamma, 1); ...
        repmat(1 / h_alpha, 2, 1); 1 / h_0])) \ all_cov_train' * W_save(:, i);
    my_pred = my_pred + all_cov_test * coef_hat / nmc;
    Ynewhat = all_cov_test * coef_hat;
    for j = 1:cutnum
        survStud = 1 - tcdf((ycutsBS(j) - Ynewhat) / sqrt(b_0 / a_0), 2 * a_0);
        SurvFunc(:, j) = SurvFunc(:, j) + survStud / nmc;
    end
end

% Compare estimates to true times. Note that est is really log of times.
obs_time = log(t_star_test);
scatter(log(t_star_test), my_pred)
hold on;
scatter(obs_time(find(delta_test)), my_pred(find(delta_test)), 6, 'red');
hold off;

% Evaluate prediction accuracy
Cindex_joint = Cindex(exp(my_pred), t_star_test, delta_test);
fprintf('C Index for joint model was %g\n', Cindex_joint);

% Get integrated Brier score
[BS, IBS] = BrierScore(ycutsBS, log(t_star_test), delta_test, SurvFunc);
fprintf('IBS for joint model was %g\n', IBS);

% Now do the same process for model without graph
b = 0;
pii = 0;

% Increase a: set it so that prior prob of var inclusion is 0.2
a = log(0.25);

% Reset seed so that this section can be run separately
rng(398023);
tic
[gamma_save_no_graph, Omega_save_no_graph, adj_save_no_graph, ar_gamma_no_graph, info_no_graph, W_save_no_graph] = ...
    MCMC_LM_scalable_AFT(X_train, ...
    t_star_train, delta_train, Z_train, ...
    a_0, b_0, h_0, h_alpha, h_beta, a, b, v0, v1, lambda, pii, ...
    gamma_init, burnin, nmc, true);
toc

csvwrite(strcat('./Output/mh_info_GBM_a', num2str(a), '_b_', ...
    num2str(b), '.csv'), ...
    [info_no_graph.n_add_prop', info_no_graph.n_add_accept' ...
    info_no_graph.n_remove_prop', info_no_graph.n_remove_accept']);
csvwrite(strcat('./Output/full_gamma_GBM_a', num2str(a), '_b_', ...
    num2str(b), '.csv'), info_no_graph.full_gamma');
csvwrite(strcat('./Output/gamma_save_GBM_a', num2str(a), '_b_', ...                                   
                 num2str(b), '.csv'), gamma_save_no_graph);
csvwrite(strcat('./Output/W_save_GBM_a', num2str(a), '_b_', ...         
		num2str(b), '.csv'), W_save_no_graph);

% Save plot of MCMC performance
trace_plot = plot(1:(burnin + nmc), sum(info_no_graph.full_gamma, 1))
xlabel('Iteration')
ylabel('Number of variables')
title('Blue = total selections')
saveas(trace_plot, strcat('./Output/MCMC_traceplot_GBM_no_graph_train', num2str(n_train), '_a', ...
    num2str(a), '_b', num2str(b), '.png'), 'png');

ppi_var = mean(gamma_save_no_graph, 2);

% Get true and false positive rates for variable selection
sel_var = ppi_var > 0.5;
fprintf('Number of selected variables without MRF = %g\n', sum(sel_var));
fprintf('Variable selections:');
find(sel_var)

% Compute prediction as MCMC avg prediction given var selection
my_pred = zeros(n_test, 1);
SurvFunc = zeros(n_test, cutnum);
for i = 1:nmc
    p_gamma = sum(gamma_save_no_graph(:, i));
    X_train_sel = X_train(:, logical(gamma_save_no_graph(:, i)));
    X_test_sel = X_test(:, logical(gamma_save_no_graph(:, i)));
    
    % Need to include fixed covariates + intercept term
    all_cov_train = [X_train_sel, Z_train, ones(n_train, 1)];
    all_cov_test = [X_test_sel, Z_test, ones(n_test, 1)];
    coef_hat = (all_cov_train' * all_cov_train + diag([repmat(1 / h_beta, p_gamma, 1); ...
        repmat(1 / h_alpha, 2, 1); 1 / h_0])) \ all_cov_train' * W_save(:, i);
    my_pred = my_pred + all_cov_test * coef_hat / nmc;
    Ynewhat = all_cov_test * coef_hat;
    for j = 1:cutnum
        survStud = 1 - tcdf((ycutsBS(j) - Ynewhat) / sqrt(b_0 / a_0), 2 * a_0);
        SurvFunc(:, j) = SurvFunc(:, j) + survStud / nmc;
    end
end


% Compare estimates to true times. Note that est is really log of times.
obs_time = log(t_star_test);
scatter(log(t_star_test), my_pred)
hold on;
scatter(obs_time(find(delta_test)), my_pred(find(delta_test)), 6, 'red');
hold off;

% Evaluate prediction accuracy
Cindex_no_graph = Cindex(exp(my_pred), t_star_test, delta_test);
fprintf('C Index for model without MRF was %g\n', Cindex_no_graph);

% Get integrated Brier score
upper_time = max(log(t_star_test(delta_test == 0)));
[BS_no_graph, IBS_no_graph] = BrierScore(ycutsBS, log(t_star_test), delta_test, SurvFunc);
fprintf('IBS for model without MRF was %g\n', IBS_no_graph);

