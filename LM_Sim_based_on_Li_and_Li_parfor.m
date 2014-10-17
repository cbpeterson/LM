% For running locally
cd 'C:\Users\Marina\Desktop\LM only code\Code'
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';
addpath 'C:\Users\Marina\Desktop\LM only code\Code\glmnet_matlab';
% if matlabpool('size') == 0
%    matlabpool(2)
% end

% Number of iterations to run at each setting for n
niter = 100;

% In case I need to debug
dbstop if error;

% Settings from Li and Li paper - run first three methods to replicate
% results in their paper
li_and_li_setting = false;
if li_and_li_setting
    % Number of transcription factors
    num_tfs = 200;
    
    % Number of genes regulated by each transcription factor
    g_per_tf = 10;
    
    % Number of genes with sign changed in models 2 and 4
    g_change = 3;
    
    input_folder = './Simulation/Inputs/FullDim/';
else
    % Number of transcription factors
    num_tfs = 40;
    
    % Number of genes regulated by each transcription factor
    g_per_tf = 5;
    
    % Number of genes with sign changed in models 2 and 4
    g_change = 2;
    
    input_folder = './Simulation/Inputs/ReducedDim/';
end

% Sample size for training and test sets
n = 100;

% Total number of genes
p = num_tfs * (g_per_tf + 1);

% Since Z's are set to 0 here, this param does not matter
h_alpha = 0;

% Prior param for G-Wishart
delta_prior = 3;

% Record performance summary
% Dim 1: 4 = models 1 to 4
% Dim 2: 4 = sens, spec, mcc, pmse
% Dim 3: 5 = lasso, en, LL, selection using graph, selection w/o graph
% Dim 4 = iter
nmodel = 4;
nmetric = 5;
nmethod = 5;
full_perf_summary = zeros([nmodel, nmetric, nmethod, niter]);

% Also keep track of lambda2 selected values for Li and Li method
% and edge selection for proposed method
best_lambda2_save = zeros(nmodel, niter);
edge_sel_perf = zeros(2, nmodel, niter); % First dim = tpr and fpr

% Number of MCMC iterations for Bayesian methods
burnin  = 10000;
nmc = 10000;

for model = 1:nmodel
    % Record performance summmary for current model
    % Dim 1: 4 = sens, spec, mcc, pmse
    % Dim 2: 5 = lasso, en, LL, selection using graph, selection w/o graph
    % Dim 3 = iter
    model_perf_summary = zeros([nmetric, nmethod, niter]);
    
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
    Adj_true = Adj_true(1:(p_true * (p_true - 1) / 2));
    
    % True model: first p_true genes contribute to outcome
    gamma_true = cat(1, ones(p_true, 1), zeros(p - p_true, 1));
    
    % Prior param for G-Wishart
    D_prior = eye(p);
    
    % Parameters of MRF prior - how to determine proper settings for a and b?
    % Li and Zhang discuss this, esp phase transition property
    
    % Approach here: fix b, then figure out values for lambda and a that
    % correspond to appropriate prior probabilities for edges and
    % variables
    b = 0.1;
    
    % c_e = desired prior probability of edges = number of edges among true
    % variables divided by total possible number of edges
    c_e = ((sum(sum(abs(Omega_true(1:p_true, 1:p_true)) > 0)) - p_true) / 2) / ...
        ( p  * (p - 1) / 2);
    lambda_mrf = -c_e / (c_e * (exp(2 * b) - 1) - exp(2 * b));
    
    % c_v = desired prior probability of variables
    c_v = p_true / p;
    a = log(c_v / (1 - c_v)) - 2 * b * p / 10;

    % Prior probability of variable inclusion for Bayesian variable selection
    lambda_bvs = p_true / p;
    
    parfor cur_iter = 1:niter
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
        
        best_lambda2_save(model, cur_iter) = best_lambda2;
        
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
        
        if ~li_and_li_setting
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
            
            % Also try calculating PMSE using median model
            p_gamma = sum(sel_nograph);
            X_sel = X(:, logical(sel_nograph));
            X_test_sel = X_test(:, logical(sel_nograph));
            beta_hat = (X_sel' * X_sel + 1 / h_beta * eye(p_gamma)) \ X_sel' * Y;
            my_pred = X_test_sel * beta_hat;
            pmse_BVS_median = mean((my_pred - Y_test) .^ 2);
            
            % Store results
            cur_perf_summary(1, 4) = tpr_nograph;
            cur_perf_summary(2, 4) = 1 - fpr_nograph;
            cur_perf_summary(3, 4) = mcc_nograph;
            cur_perf_summary(4, 4) = pmse_BVS;
            cur_perf_summary(5, 4) = pmse_BVS_median;
            
            % PROPOSED METHOD -----------------------------------------------------
            % Initial value of Omega (precision matrix)
            Omega_init = eye(p);
            
            % Run MCMC sampler for joint graph and variable selection
            % Clinical covariates Z are set to all zeros here
            % Since p is large, param summary_only is set to true            
            tic
            [gamma_save, Omega_save, adj_save, ar_gamma, info] = MCMC_LM_GWishart_simplified(X, Y, zeros(n, 5), ...
                a_0, b_0, h_alpha, h_beta, a, b, lambda_mrf, delta_prior, D_prior, ...
                gamma_init, Omega_init, burnin, nmc, true);
            toc
            
            % Collect additional performance information for proposed
            % method
            csvwrite(strcat('./Output/mh_info_model', num2str(model), '_iter', ...
                num2str(cur_iter), '.csv'), ...
                [info.n_add_disc_prop', info.n_add_disc_accept' ...
                info.n_remove_disc_prop', info.n_remove_disc_accept', ...
                info.n_add_conn_prop', info.n_add_conn_accept' ...
                info.n_remove_conn_prop', info.n_remove_conn_accept', ...
                info.n_no_prop']);
            csvwrite(strcat('./Output/full_gamma__model', num2str(model), '_iter', ...
                num2str(cur_iter), '.csv'), info.full_gamma');
            csvwrite(strcat('./Output/node_degrees_model', num2str(model), '_iter', ...
                num2str(cur_iter), '.csv'), info.node_degrees');
            
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
            saveas(h, strcat('./Output/MCMC_traceplot_model', num2str(model), '_iter', ...
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
            
            edge_sel_perf(:, model, cur_iter) = [tpr_edges, fpr_edges];
            
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
            
            % Also try calculating PMSE using median model
            p_gamma = sum(sel_var);
            X_sel = X(:, logical(sel_var));
            X_test_sel = X_test(:, logical(sel_var));
            beta_hat = (X_sel' * X_sel + 1 / h_beta * eye(p_gamma)) \ X_sel' * Y;
            my_pred = X_test_sel * beta_hat;
            my_pmse_median = mean((my_pred - Y_test) .^ 2);
            
            % Store results
            cur_perf_summary(1, 5) = tpr_var;
            cur_perf_summary(2, 5) = 1 - fpr_var;
            cur_perf_summary(3, 5) = mcc_var;
            cur_perf_summary(4, 5) = my_pmse;
            cur_perf_summary(5, 5) = my_pmse_median;
        end
        
        % Record performance for current iteration
        model_perf_summary(:, :, cur_iter) = cur_perf_summary;
    end

    csvwrite(strcat('Perf_Li_and_Li_model', num2str(model), '.csv'), model_perf_summary);
    full_perf_summary(model, :, :, :) = model_perf_summary;
end

% matlabpool close

% Write full values to file
csvwrite('./Output/Perf_Li_and_Li_sim.csv', full_perf_summary);
csvwrite('./Output/Li_and_Li_best_lambda2.csv', best_lambda2_save);
csvwrite('./Output/LM_graph_edge_sel_perf.csv', edge_sel_perf);

% Prepare table 1 as in Li and Li paper
table1 = zeros(nmodel * 2, nmethod * nmetric);
for model = 1:nmodel
    row = (model - 1) * 2 + 1;
    for measure = 1:nmetric % sens, spec, mcc and PMSE
        for method = 1:nmethod
            col = (measure - 1) * nmethod + method;
            table1(row, col) = mean(squeeze(full_perf_summary(model, measure, method, :)));
            table1(row + 1, col) = std(squeeze(full_perf_summary(model, measure, method, :)));
        end
    end
end
csvwrite('./Output/Li_and_Li_table1_summary.csv', table1);

% Look at relationship between Lambda2 values and sens, spec, mcc, and PMSE
% for each model in terms of diff from EN
sel1 = [best_lambda2_save(1, :)', squeeze(full_perf_summary(1, :, 2, :))' - squeeze(full_perf_summary(1, :, 3, :))'];
sel2 = [best_lambda2_save(2, :)', squeeze(full_perf_summary(2, :, 2, :))' - squeeze(full_perf_summary(2, :, 3, :))'];
sel3 = [best_lambda2_save(3, :)', squeeze(full_perf_summary(3, :, 2, :))' - squeeze(full_perf_summary(3, :, 3, :))'];
sel4 = [best_lambda2_save(4, :)', squeeze(full_perf_summary(4, :, 2, :))' - squeeze(full_perf_summary(4, :, 3, :))'];

% Make scatter plots to look at diff in PMSE from EN to LL
% Larger positive values are good
scatter(log(sel1(:, 1)), sel1(:, 4))
scatter(log(sel2(:, 1)), sel2(:, 4))
scatter(log(sel3(:, 1)), sel3(:, 4))
scatter(log(sel4(:, 1)), sel4(:, 4))