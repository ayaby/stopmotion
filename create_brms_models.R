#!/usr/bin/env Rscript --vanilla

#
# This code is a slight modification of code written by Alex Quent:
#  https://github.com/JAQuent/bayesianMLM/tree/master/CBU_clusterGuide
#

library(brms)
library(rslurm)

# Sets seed to arbitrary value  for reproducibility
set.seed(26)

# Gets the brms directory from the command line arguments. If none is provided, runs in current directory
args = commandArgs(trailingOnly=TRUE)
orig_wd <- getwd()
if (length(args)==1) {
  # Switches to the brms directory if one was provided
  setwd(args[1])
} else if (length(args)>1) {
  stop("At most one argument should be provided")
}

# General brms params
n_runs <- 8
iter_per_chain <- 4000
n_nodes <- 1

# Reads dataframe, assumes it is in the provided directory
data_df <- read.csv("brms_df.csv")

# Turns the relevant fields into factors/logical/numeric and scales the only numeric predictor
data_df[,"subj"] <- as.factor(data_df[,"subj"])
data_df[,"non_foil_q"] <- as.factor(data_df[,"non_foil_q"])
data_df[,"action_idx"] <- scale(as.numeric(data_df[,"action_idx"]))
data_df[,"ansOld"] <- as.logical(data_df[,"ansOld"])
data_df[,"q_type"] <- as.factor(data_df[,"q_type"])
data_df[,"surprise_type"] <- as.factor(data_df[,"surprise_type"])
data_df[,"group"] <- as.factor(data_df[,"group"])
data_df[,"exc_subj"] <- as.logical(data_df[,"exc_subj"])

# Gets subset of data for proactive analysis, removing excluded subjects from analysis
data_df <- subset(data_df, (q_type=="preT" | q_type=="postT") & exc_subj==FALSE)


# Creates lists of model names and the matching formulas (the code will then iterate over these). These include
# models with/without each interaction/factor that will later be tested, so evidence can be assessed by
# model comparison with the bayes_factor function
model_names <- c("pro_full", "pro_no_3way", "pro_no_SQT", "pro_w_S", "pro_no_S",
                 "HC_pro_full", "HC_pro_no_3way", "HC_pro_no_SQT", "HC_pro_w_S", "HC_pro_no_S")

model_formulas <- c(
  "ansOld ~ foil*(surprise_type*group*q_type + action_idx) + (1|subj) + (1|non_foil_q)",
  paste0("ansOld ~ surprise_type*group*q_type + foil*(surprise_type*group + group*q_type +",
         "surprise_type:q_type + action_idx) + (1|subj) + (1|non_foil_q)"),
  paste0("ansOld ~ surprise_type*group*q_type + foil*(surprise_type*group + group*q_type +",
         " action_idx) + (1|subj) + (1|non_foil_q)"),
  paste0("ansOld ~ surprise_type*group*q_type + foil*(surprise_type + group*q_type + action_idx)",
         " + (1|subj) + (1|non_foil_q)"),
  "ansOld ~ surprise_type*group*q_type + foil*(group*q_type + action_idx) + (1|subj) + (1|non_foil_q)",
  "HC_ansOld ~ foil*(surprise_type*group*q_type + action_idx) + (1|subj) + (1|non_foil_q)",
  paste0("HC_ansOld ~ surprise_type*group*q_type + foil*(surprise_type*group + group*q_type +",
         "surprise_type:q_type + action_idx) + (1|subj) + (1|non_foil_q)"),
  paste0("HC_ansOld ~ surprise_type*group*q_type + foil*(surprise_type*group + group*q_type +",
         " action_idx) + (1|subj) + (1|non_foil_q)"),
  paste0("HC_ansOld ~ surprise_type*group*q_type + foil*(surprise_type + group*q_type + action_idx)",
         " + (1|subj) + (1|non_foil_q)"),
  "HC_ansOld ~ surprise_type*group*q_type + foil*(group*q_type + action_idx) + (1|subj) + (1|non_foil_q)")


# Defines the family and link function used for the models
model_family <- "bernoulli(link='logit')"

# Defines priors for the intercept and factors
model_priors <- c(prior(student_t(1, 0, 10), class = "Intercept"), 
                  prior(student_t(5, 0, 2.5), class = "b"))

# Loops over models and for each creates a starter model, defines helper function to add chains to
# an existing model and creates a batch script (which is not run)
for (m in 1:length(model_names)) {
  eval(parse(text=gsub("<formula>", model_formulas[m], gsub("<family>", model_family, 
      gsub("<model_name>", model_names[m], "
                                                                 
        # Random seeds for the model
        pars_<model_name>  <- data.frame(i = 1:n_runs, seed = sample(99999, n_runs))
                                                                 
        # Defines the starter model
        starter_<model_name> <- brm(<formula>, data = data_df, family = <family>, save_all_pars = TRUE, 
                            cores = 1, chains=1, iter=iter_per_chain, sample_prior = TRUE, save_dso = TRUE, 
                            seed = pars_<model_name>[1, 'seed'], prior=model_priors, control = list(max_treedepth = 15))
                                                                 
        # Defines a helper function and a job script for each model that will combine the different runs
        helper_<model_name><- function(i, seed){
          if(i == 1){
            return(list(i = i, seed = seed, model = starter_<model_name>))
          } else {
            model_<model_name> <- update(starter_<model_name>, newdata = data_df, recompile = FALSE, 
                              cores = 1, chains = 1, iter = iter_per_chain, save_all_pars = TRUE, 
                              sample_prior = TRUE, save_dso = TRUE, seed = seed)
            return(list(i = i, seed = seed, model = model_<model_name>))
          }
        }
        sjob_<model_name> <- slurm_apply(helper_<model_name>, pars_<model_name>, jobname = 'model_<model_name>',
        add_objects = c('starter_<model_name>', 'data_df', 'model_priors', 'iter_per_chain'),
        nodes = n_nodes, cpus_per_node = n_runs)
      ")))))
}

# Returns to the previous working directory
setwd(orig_wd)

