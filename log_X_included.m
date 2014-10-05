function [log_p] = log_X_included(X_incl, Omega_incl)
  % TODO: consider more efficient way to update this using Sig rather than
  % recomputing full matrix inverse
  % Also consider trying to use version of mvnpdf which is already on log
  % scale (not part of basic stats toolbox)

  p_gamma = size(X_incl, 2);
  Sigma = inv(Omega_incl);
  
  if p_gamma > 0
      log_p = sum(log(mvnpdf(X_incl, zeros(1, p_gamma), Sigma)));
  else
      log_p = 0;
  end
end
