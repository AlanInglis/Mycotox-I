
#  libraries
library(keras3)
library(tensorflow)
library(reticulate)
#################
# Setting seeds #
#################

#  R seed
set.seed(1701)
# Set backend
use_backend("tensorflow")

# Python seed control
np <- import("numpy", convert = TRUE)
random <- import("random", convert = TRUE)
tf <- import("tensorflow", convert = TRUE)

np$random$seed(1701L)
random$seed(1701L)
tf$random$set_seed(1701L)

#  force GPU/CPU consistency/determinism
tf$config$experimental$enable_op_determinism()


#####################################################
# Pretrain autoencoder on x (predictor matrix only) #
#####################################################


# — 1. Input dimensions —

input_dim <- ncol(x)  # number of predictors


# — 2. Build encoder —

input_layer <- layer_input(shape = input_dim, name = 'autoencoder_input')

encoded <- input_layer |>
  layer_dense(units = 512, activation = 'relu') |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 256, activation = 'relu') |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 128, activation = 'relu', name = 'latent_space')

# — 3. Build decoder —

decoded <- encoded |>
  layer_dense(units = 256, activation = 'relu') |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 512, activation = 'relu') |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = input_dim, activation = 'linear', name = 'reconstructed_output')

# — 4. Combine into autoencoder model —

autoencoder <- keras_model(inputs = input_layer, outputs = decoded)

autoencoder |> compile(
  loss = 'mse',
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = list('mse')
)

# — 5. Fit autoencoder —

history <- autoencoder |> fit(
  x,
  x,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 2,
  shuffle = FALSE,
  callbacks = list(
    callback_early_stopping(patience = 10, restore_best_weights = TRUE)
  )
)

# — 6. Extract and save encoder —

encoder <- keras_model(inputs = input_layer, outputs = encoded)

# Save encoder weights to disk
#dir.create('pretrained_model', showWarnings = FALSE)
#encoder |> save_model('pretrained_model/encoder_pretrained.h5')

###############################################
## Transfer learning using pretrained encoder #
###############################################

# — 1. Reload encoder —

#encoder <- load_model('pretrained_model/encoder_pretrained.h5')

# Optional: Freeze encoder initially

freeze_weights(encoder)

# — 2. Attach task-specific heads —

encoded_output <- encoder$output

#  Regression head for continuous toxins

cont_out <- encoded_output |>
  layer_dense(units = ncol(y_cont), activation = 'relu', name = 'cont_out')

# Classification head for binary toxin detection

bin_out <- encoded_output |>
  layer_dense(units = ncol(y_bin), activation = 'sigmoid', name = 'bin_out')

# Full model

model <- keras_model(inputs = encoder$input, outputs = list(cont_out, bin_out))

#— 3. Compile model with your custom masked losses —

#unfreeze_weights(encoder)


model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = list(cont_out = masked_mse, bin_out = masked_bce),
  metrics = list(cont_out = masked_mse, bin_out = masked_bce)
)

# — 4. Fit the model —

early_stop <- callback_early_stopping(
  monitor = 'val_loss',
  patience = 25,
  restore_best_weights = TRUE
)

history <- model |> fit(
  x_train,
  list(cont_out = y_cont_train, bin_out = y_bin_train),
  epochs = 500,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(early_stop),
  shuffle = FALSE,
  verbose = 2
)

# — 5. Evaluate on test set —

eval <- model |> evaluate(
  x_test,
  list(cont_out = y_cont_test, bin_out = y_bin_test),
  verbose = 0
)
print(eval)

#saveRDS(eval, "eval_freeze_model.rds")
#saveRDS(eval, "eval_unfreeze_model.rds")

########################################################################
##  Test-set metrics: continuous + binary responses  ##################

library(purrr)
library(tibble)
library(pROC) 
library(dplyr)

## 1.  Predictions -----------------------------------------------------
pred <- model |>
  predict(x_test)

y_cont_pred <- pred[[1]]    # matrix  n_test × 21
y_bin_pred  <- pred[[2]]     # matrix  n_test × 21

## 2.  Helpers ---------------------------------------------------------
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
  
  # One-class case → Accuracy (all predictions of that class) and AUC = 0.5
  acc <- mean(pred_ok == obs_ok)
  c(F1_or_Acc = acc, AUC = 0.5)
}


## 3.  Assemble tidy table --------------------------------------------
cont_tbl <- map2_dfc(
  as.data.frame(y_cont_test),
  as.data.frame(y_cont_pred),
  metrics_cont
) |>
  t() |>
  as_tibble(rownames = "Variable") |>
  rename(RMSE = V1, R2 = V2)



bin_tbl <- map2_dfc(
  as.data.frame(y_bin_test),
  as.data.frame(y_bin_pred),
  metrics_bin
) |>
  t() |>
  as_tibble(rownames = "Variable") |>
  rename(F1_or_Acc = V1, AUC = V2) |>
  mutate(Variable = str_remove(Variable, "_bin$"))



metrics_tbl <- cont_tbl |>
  left_join(bin_tbl, by = "Variable") |>
  arrange(Variable)


print(metrics_tbl, n = Inf)

# save as rds
#saveRDS(metrics_tbl, "mycotoxin_joint_metrics_freeze.rds")
#saveRDS(metrics_tbl, "mycotoxin_joint_metrics_unfreeze.rds")






















