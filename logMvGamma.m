function [ log_res ] = logMvGamma(a, p)
  log_res = p * (p - 1) / 4 * log(pi);
  for j = 1:p
      log_res = log_res + gammaln(a + (1-j) / 2);
  end
end