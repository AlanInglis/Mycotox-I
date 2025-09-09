########################################################################
## Multi-task network: joint continuous + binary mycotoxin prediction ##
##                                                                    ##
########################################################################
## Aim – For every toxin we predict                                   ##
##   • its observed concentration (regression)                        ##
##   • a 0 / 1 indicator of “present above zero” (classification)     ##
## Missing responses (NA / NaN) are masked out for both tasks.        ##
########################################################################
## Note – The underlying dataset is private and not shared here.      ##
##        This script is provided to illustrate the methods and       ##
##        modelling workflow used in the Mycotox-I project.           ##
########################################################################
## Script outline                                                     ##
##   0. Setup: libraries, seeds, TensorFlow determinism               ##
##   1. Data import: load dataset                                     ##
##   2. Predictors & responses: define variables                      ##
##   3. Train/test split                                              ##
##   4. Masked losses: custom MSE & BCE ignoring missing values       ##
##   5. Model: shared layers + dual outputs (continuous & binary)     ##
##   6. Fit: training with early stopping                             ##
##   7. Evaluate: performance on test set                             ##
##   8. Predictions & metrics: RMSE, R², F1, AUC tables               ##
########################################################################

######################## 0.  Setup #####################################
library(tidyverse)
library(keras3)
library(reticulate)

# Set up keras3 with tensorflow backend
tf <- import("tensorflow")
set.seed(1701)
keras3::use_backend("tensorflow")
# Import Python modules
np <- import("numpy", convert = TRUE)
tf <- import("tensorflow", convert = TRUE)
random <- import("random", convert = TRUE)

# Set seeds for Python, NumPy, and TensorFlow
np$random$seed(1701L)
random$seed(1701L)
tf$random$set_seed(1701L)
# forece determinism
tf$config$experimental$enable_op_determinism()

######################## 1.  Data import #################################

# load csv
dat <- readRDS("dat_nn.rds")

######################## 2. Predictors and response ######################

# remove replicate column
dat <- dat %>%
  select(-replicate_number)

# response variable names
start_resp   <- match("Trichothence_producer", names(dat))   # first response
resp_cols    <- names(dat)[start_resp:length(dat)]            # all responses
# continuous and binary response variable names
bin_vars <- resp_cols[str_detect(resp_cols, "_bin$")]   # binary
cont_vars <- setdiff(resp_cols, bin_vars)           # continuous
# predictor variable names
pred_vars <- setdiff(names(dat), resp_cols)

# turn into matrix
dat <- as.matrix(dat)

# separate predictors from responses
x <- dat[,pred_vars]
y_cont <- dat[, cont_vars]
y_bin  <- dat[, bin_vars]


######################## 3.  Train : test split ########################
set.seed(1701)

# sample 80% of rows for training
idx <- sample(nrow(x), 0.8 * nrow(x))

# split into train and test sets
x_train     <- x[idx, ] # training predictors
x_test      <- x[-idx, ] # test predictors
y_cont_train<- y_cont[idx, ] # training continuous responses
y_cont_test <- y_cont[-idx, ] # test continuous responses
y_bin_train <- y_bin[idx, ] # training binary responses
y_bin_test  <- y_bin[-idx, ] # test binary responses


######################## 4.  Masked losses #############################

# Custom loss functions that ignore NA values in y_true
# For continuous responses: masked MSE
masked_mse <- function(y_true, y_pred) {
  # mask: TRUE where the target is observed
  keep        <- tf$math$logical_not(tf$math$is_nan(y_true)) # keep is a boolean tensor
  keep_f32    <- tf$cast(keep, tf$float32) # keep_f32 is a float tensor
  
  y_true_safe <- tf$where(keep, y_true, tf$zeros_like(y_true)) # y_true_safe has no NaNs
  
  se          <- tf$math$square(y_pred - y_true_safe) * keep_f32 # squared errors, masked
  
  tf$math$reduce_sum(se) /
    (tf$math$reduce_sum(keep_f32) + 1e-7) # avoid division by zero
}

# For binary responses: masked binary cross-entropy
masked_bce <- function(y_true, y_pred) {
  keep        <- tf$math$logical_not(tf$math$is_nan(y_true)) 
  keep_f32    <- tf$cast(keep, tf$float32)
  
  y_true_safe <- tf$where(keep, y_true, tf$zeros_like(y_true))
  
  # clip logits so log(...) is finite
  y_pred_clip <- tf$clip_by_value(y_pred, 1e-7, 1 - 1e-7)
  
  ce_elem <- -(y_true_safe * tf$math$log(y_pred_clip) +
                 (1 - y_true_safe) * tf$math$log(1 - y_pred_clip)) # element-wise cross-entropy
  
  ce_masked <- ce_elem * keep_f32 # masked cross-entropy
  
  tf$math$reduce_sum(ce_masked) /
    (tf$math$reduce_sum(keep_f32) + 1e-7) # avoid division by zero
}

######################## 5.  Model #####################################

# Define the model architecture
input_dim  <- ncol(x_train) # number of predictors
output_dim <- length(cont_vars) # number of continuous (or binary) responses

# Input layer
inputs <- layer_input(shape = input_dim, name = "predictors")

# Shared hidden layers
shared <- inputs |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dropout(rate = 0.20) |>
  layer_dense(units = 64,  activation = "relu") |>
  layer_dropout(rate = 0.20)

# Output layers for continuous and binary responses
cont_out <- shared |>
  layer_dense(units = output_dim, activation = "relu", name = "cont_out") 

bin_out  <- shared |>
  layer_dense(units = output_dim, activation = "sigmoid", name = "bin_out")

# Define the model with two outputs
model <- keras_model(inputs = inputs, outputs = list(cont_out, bin_out))

# Compile the model with custom masked loss functions
model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss      = list(cont_out = masked_mse,
                   bin_out  = masked_bce),
  metrics   = list(cont_out = masked_mse,
                   bin_out  = masked_bce)
)


######################## 6.  Fit #######################################

# Early stopping callback
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 25,
  restore_best_weights = TRUE
)


# Fit the model
history <- model |>
  fit(
    x_train,
    list(cont_out = y_cont_train, bin_out = y_bin_train),
    epochs           = 500,
    batch_size       = 32,
    validation_split = 0.2,
    callbacks        = list(early_stop),
    verbose          = 2,
    shuffle = FALSE
  )

######################## 7.  Evaluate ##################################

# Evaluate the model on the test set
eval <- model |>
  evaluate(x_test,
           list(cont_out = y_cont_test, bin_out = y_bin_test),
           verbose = 0)
print(eval)


######################## 8.  Predictions and metrics ###################

# Load libraries
library(purrr)
library(tibble)
library(pROC) 
library(dplyr)

# Make predictions on the test set
pred <- model |>
  predict(x_test)

# Extract continuous and binary predictions
y_cont_pred <- pred[[1]]  # continuous
y_bin_pred  <- pred[[2]]  # binary   

# give the columns names
colnames(y_cont_pred) <- cont_vars
colnames(y_bin_pred)  <- bin_vars


# Define functions to compute metrics for continuous and binary outcomes

# Continuous metrics: RMSE and R²
metrics_cont <- function(obs, pred) {
  ok <- !is.na(obs)
  n  <- sum(ok)
  if (n == 0)
    return(c(RMSE = NA_real_, R2 = NA_real_))
  
  resid <- pred[ok] - obs[ok]
  rmse  <- sqrt(mean(resid^2))
  
  r2 <- if (n > 1 && var(obs[ok]) > 0) {
    ss_tot <- sum((obs[ok] - mean(obs[ok]))^2)
    1 - sum(resid^2) / ss_tot
  } else {
    NA_real_
  }
  
  c(RMSE = rmse, R2 = r2)
}

# Binary metrics: F1 score (or accuracy if one class) and AUC
metrics_bin <- function(obs, prob, threshold = 0.5) {
  ok <- !is.na(obs)
  if (sum(ok) == 0)
    return(c(F1_or_Acc = NA_real_, AUC = NA_real_))
  
  obs_ok  <- obs[ok]
  prob_ok <- prob[ok]
  pred_ok <- as.numeric(prob_ok >= threshold)
  
  # Two-class case  → F1 and AUC
  if (length(unique(obs_ok)) == 2) {
    tp <- sum(pred_ok == 1 & obs_ok == 1)
    fp <- sum(pred_ok == 1 & obs_ok == 0)
    fn <- sum(pred_ok == 0 & obs_ok == 1)
    
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1        <- ifelse(precision + recall == 0, 0,
                        2 * precision * recall / (precision + recall))
    
    auc <- as.numeric(pROC::auc(obs_ok, prob_ok))
    
    return(c(F1_or_Acc = f1, AUC = auc))
  }
  
  # One-class case: Accuracy (all predictions of that class) and AUC = 0.5
  acc <- mean(pred_ok == obs_ok)
  c(F1_or_Acc = acc, AUC = 0.5)
}


# Compute metrics for all continuous and binary variables
cont_tbl <- map2_dfc(
  as.data.frame(y_cont_test),
  as.data.frame(y_cont_pred),
  metrics_cont
) |>
  t() |>
  as_tibble(rownames = "Variable") |>
  rename(RMSE = V1, R2 = V2)



# table of continuous metrics
bin_tbl <- map2_dfc(
  as.data.frame(y_bin_test),
  as.data.frame(y_bin_pred),
  metrics_bin
) |>
  t() |>
  as_tibble(rownames = "Variable") |>
  rename(F1_or_Acc = V1, AUC = V2) |>
  mutate(Variable = str_remove(Variable, "_bin$"))


# Join continuous and binary metrics into one table
metrics_tbl <- cont_tbl |>
  left_join(bin_tbl, by = "Variable") |>
  arrange(Variable)


# View the metrics table
metrics_tbl

