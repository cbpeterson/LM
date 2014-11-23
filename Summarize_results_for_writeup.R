# Script to process output files from Matlab simulations
setwd("/Users/cbp/Dropbox/Research/PhD research/Graphical models/Linear model setting/Linear model only/Code/branch3 - sample full graph")

# Number of models
nmodel <- 4

# sens, spec, mcc and PMSE
nmetric <- 4

# Number of methods to compare
nmethod <- 5

# Number of iterationsn run
niter <- 50

# Prepare table 1 as in Li and Li paper
table1 <- matrix(0, nrow = nmodel * 2, ncol = nmethod * nmetric)

# Get accuracy of graph structure learning
graph_tpr <- rep(NA, niter)
graph_fpr <- rep(NA, niter)

for (model in 1:nmodel) {
  cur_row <- (model - 1) * 2 + 1
  
  # TODO: finalize which parameter combination is best
  if (model == 1 || model == 2) {
    a = 3
  } else {
    a = 2.3
  }
    
  # Read in output files for curent model across all iterations
  res_cur_model <- array(0, dim = c(nmetric, nmethod, niter))
  for (iter in 1:niter) {
    res_cur_model[ , , iter] <- as.matrix(read.csv(paste("Output/Model", model, "/a_", a , "/perf_summary_", model, "_iter", iter,
                                                         ".csv", sep = ""), header = FALSE))
    graph_res <- as.matrix(read.csv(paste("Output/Model", model, "/a_", a , "/perf_edges_", model, "_iter", iter,
                                          ".csv", sep = ""), header = FALSE))
    graph_tpr[iter] <- graph_res[1]
    graph_fpr[iter] <- graph_res[2]
  }
  
  for (measure in 1:nmetric) {
    for (method in 1:nmethod) {
      cur_col <- (measure - 1) * nmethod + method
      table1[cur_row, cur_col] <- mean(res_cur_model[measure, method, ])
      table1[cur_row + 1, cur_col] <- sd(res_cur_model[measure, method, ]) / sqrt(niter)
    }
  }
}

# Add row/col ids
col_names <- paste(c(rep("Sens", nmethod), rep("Spec", nmethod), rep("MCC", nmethod), rep("PMSE", nmethod)),
                rep(c("Lasso", "EN", "LL", "BVS", "Joint"), nmetric), sep = "_")
row_names <- rep(c("Mean", "SE"), nmodel)
table1 <- data.frame(table1)
names(table1) <- col_names

round(table1[ , c(1:10,16:20)], 3)

# Mean mean and sd of graph structure learning performance
mean(graph_tpr)
sd(graph_tpr)
mean(graph_fpr)
sd(graph_fpr)

# What is average node degree and overall sparsity?
p <- 240
n_true_edges <- 5 * 40
n_tp <- n_true_edges * mean(graph_tpr)
n_missing_edges <- p * (p - 1) / 2 - n_true_edges
n_false_edges <- n_missing_edges * mean(graph_fpr)
tot_edges_inferrred <- n_true_edges + n_false_edges

# Average node degree
tot_edges_inferrred * 2 / p

# Average sparsity
tot_edges_inferrred / p / (p-1) * 2

# Sensitivity analysis ----------------------------------------------------------------------------------------------------------------

# Look at difference in number of selected variables and also other metrics for changing values of a
n_setting <- 6
sens_res <- data.frame(a_setting = 1:n_setting, a_value = c(-3.5, -3.25, -3, -2.75, -2.5, -2.25), 
                       n_vars = rep(NA, n_setting), sens = rep(NA, n_setting), spec = rep(NA, n_setting),
                       PMSE = rep(NA, n_setting))
p_true <- 24

# These were all run for model 1 using b = 0.5
model <- 1

setwd("/Users/cbp/Dropbox/Research/PhD research/Graphical models/Linear model setting/Linear model only/Code/branch3 - sample full graph/Output/Sensitivity")

for (a_setting in 1:n_setting) {

  # Read in output files for curent model across all iterations
  res_cur_setting <- array(0, dim = c(nmetric, nmethod, niter))
  for (iter in 1:niter) {
    res_cur_model[ , , iter] <- as.matrix(read.csv(paste("a_setting", a_setting, "/perf_summary_", model, "_iter", iter,
                                                         ".csv", sep = ""), header = FALSE))
  }
  
  # TPR/sensitivity
  sens_res$sens[a_setting] <- mean(res_cur_model[1, 5, ])
  
  # Specificity
  sens_res$spec[a_setting] <- mean(res_cur_model[2, 5, ])
  
  # Avg number of variables selected
  sens_res$n_vars[a_setting] <- sens_res$sens[a_setting] * p_true + (1 - sens_res$spec[a_setting]) * (p - p_true)
  
  # PMSE
  sens_res$PMSE[a_setting] <- mean(res_cur_model[4, 5, ])
}

plot(sens_res$a_value, sens_res$PMSE, pch = 19)
plot(sens_res$a_value, sens_res$sens, pch = 19)
plot(sens_res$a_value, sens_res$spec, pch = 19)

library(ggplot2)
ggplot(aes(x = sens_res$a_value, y = sens_res$n_vars), data = sens_res) + geom_point(color = "#4C0099") +
  geom_line(color = "#4C0099") + theme_bw() + xlab("Value of a") + ylab("Average number of selected variables") +
  ggtitle("Sensitivity to Markov random field parameter a")


