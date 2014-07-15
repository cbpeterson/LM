% Script to generate data for simulations
% I did this so that I could tweak various methods and get standard
% comparison to previous results

% For running locally
cd 'C:\Users\Marina\Desktop\LM only code\Code';
addpath 'C:\Research\Graphical models\Papers\Wang Li 2012 with code\Code - original download';

niter = 100;

% Settings from Li and Li paper
li_and_li_setting = true;
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

nmodel = 4;
for model = 1:nmodel
    
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
    
    % Check graph structure and partial correlation
    Omega1 = inv(Sigma(1:11, 1:11));
    par_corr = -Omega1(1,2) / sqrt(Omega1(1,1)) / sqrt(Omega1(2,2));
    
    for iter = 1:niter
        % Simulate gene data
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
        
        % Center Y
        Y = Y - repmat(mean(Y, 1), n, 1);
        
        % Create test data set
        X_test = zeros(n, p);
        for i = 1:n
            X_test(i, :) = rMNorm(zeros(p, 1), Sigma, 1)';
        end
        
        % Standardize X_test
        X_test = zscore(X_test);
        
        % Generate response variable Y with variance tau_sq
        Y_test = zeros(n, 1);
        for i = 1:n
            Y_test(i) = normrnd(X_test(i, :) * beta, sqrt(tau_sq));
        end
        
        % Center Y_test
        Y_test = Y_test - repmat(mean(Y_test, 1), n, 1);
        
        csvwrite(strcat(input_folder, 'X_', num2str(iter), '_model', num2str(model), '.csv'), X);
        csvwrite(strcat(input_folder, 'X_test_', num2str(iter), '_model', num2str(model), '.csv'), X_test);
        csvwrite(strcat(input_folder, 'Y_', num2str(iter), '_model', num2str(model), '.csv'), Y);
        csvwrite(strcat(input_folder, 'Y_test_', num2str(iter), '_model', num2str(model), '.csv'), Y_test);
    end
end