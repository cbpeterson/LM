function [ Rho ] = my_parcorr(S)
  % Compute matrix of partial correlations for given S matrix
  
  Omega = inv(S);
  p = size(S, 1);
  Rho = zeros(p, p);
  for i = 1:p
      for j = 1:p
          if i == j
              Rho(i, j) = 1;
          else
              Rho(i, j) = - Omega(i, j) / sqrt(Omega(i, i) * Omega(j, j));
          end
      end
  end
end