function[Brier, IBrier] = BrierScore(ycutsBS, ytest, deltatest, SurvFunc)

% BrierScore computes the Brier function and the integrated Brier score.
% It gives also the estimated survivor functions

%----------INPUTS----------
% ycutsBS      : The time points to estimate the survivor functions on log scale
% ytest        : log of observed times for test set
% deltatest    : censoring indicators for test set
% ypred        : predicted failure times on log scale

IBrier = 0;
Ttest = exp(ytest);
n_test = length(Ttest);
t0 = 0;
Brier = zeros(length(ycutsBS), 1);
for w = 1:length(ycutsBS)
    t = exp(ycutsBS(w));
    
    for i = 1:n_test
        
        % All failure times were observed
        if mean(deltatest) == 1
            Brier(w) = Brier(w) + ((Ttest(i) > t) - SurvFunc(i,w))^2;
        else
            % Subject i died before time t
            if ((Ttest(i) <= t)&& (deltatest(i) == 1))
                KM1 = KaplanMeir(Ttest, 1 - deltatest, Ttest(i));
                Brier(w) = Brier(w) + (SurvFunc(i,w)^2 / KM1);
            % Subject i survived beyond time t
            elseif (Ttest(i) > t)
                KM = KaplanMeir(Ttest, 1 - deltatest, t);
                Brier(w) = Brier(w) + (((1 - SurvFunc(i,w))^2) / KM);
            else
            end
        end
    end
    
    IBrier = IBrier + Brier(w) * (t - t0);
    t0 = t;
end

Brier = Brier / n_test;
IBrier = IBrier / (n_test * max(exp(ycutsBS)));

% Function to compute the Kaplan Meir estimate
function[KM] = KaplanMeir(T, delta, t)
KM = 1;
[f,x] = ecdf(T, 'censoring', 1-delta, 'function', 'survivor');
NT = length(x);
if (t < x(1))
    KM = 1;
elseif (t >= x(NT))
    KM = 0;
else
    for l=2:(NT-1)
        if (x(l) <= t) && (t < x(l+1))
            KM = f(l);
        end
    end
end

