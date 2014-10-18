function [C] = calc_C(p_gamma, b, lambda)
% Compute normalizing contstant for prior p(G_gamma|gamma)

% Number of possible edges
K = p_gamma * (p_gamma - 1) / 2;

% Note that nchoosek does not work (returns Inf) for n and k sufficiently large
% Instead compute this on the log scale via gammaln, then exponentiate

C = 0;
for m = 0:K
    C = C + exp(gammaln(K + 1) - gammaln(m + 1) - gammaln(K - m + 1) + ...
        2 * b * m + log(lambda) * m + ...
        log(1 - lambda) * (K - m));
end