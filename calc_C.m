function [C] = calc_C(p_gamma, b, lambda)
% Compute normalizing contstant for prior p(G_gamma|gamma)

% Number of possible edges
K = p_gamma * (p_gamma - 1) / 2;

% Turn off warning from nchoosek
warning('off', 'MATLAB:nchoosek:LargeCoefficient');

C = 0;
for m = 0:K
    C = C + nchoosek(K, m) * exp(2 * b * m + log(lambda) * m + ...
        log(1 - lambda) * (K - m));
end