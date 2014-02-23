function [ tpr, fpr, mcc ] = tpr_fpr_var( var_sel, gamma_true )
% Calculate true positive and false positive rates and matthew correlation
% coefficient for variable selection given estimated selection and truth

    tp = sum(var_sel & gamma_true);
    fp = sum(var_sel & ~gamma_true);
    tn = sum(~var_sel & ~gamma_true);
    fn = sum(~var_sel & gamma_true);
    
    % True positive rate and false positive rate for variable selection
    tpr = tp / (tp + fn);
    fpr = fp / (fp + tn);
    
    % Matthews correlation coefficient
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
end
