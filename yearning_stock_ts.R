library(tidyverse)
library(reticulate)
library(tensorflow)
library(tfprobability)
library(tfdatasets)
library(keras)
library(caret)
library(roll)


reticulate::use_condaenv('tf2gpu', required = TRUE)

### For Aaron's COMP #################################
## Make sure tf gpu doesn;t f up
py_config()

tf_config()

tf_gpu_configured()

md <- keras_model_sequential() %>% layer_dense(units=2, input_shape=5, activation='relu')

rm(md)

#############################
x_train <- readRDS('D:/x_train_array.RDS')
x_train_supp <- readRDS('D:/x_train_supp.RDS')
y_train_df <- readRDS('D:/y_train.RDS')

y_train <- y_train_df$close_out1/y_train_df$close_last

part <- createFolds(y_train_df$close_out1, k=3)

## Split to Train/Test/Valid
x_valid <- x_train[part[[1]],,]
x_valid_supp <- x_train_supp[part[[1]],]
y_valid <- y_train[part[[1]]]


x_test <- x_train[part[[2]],,]
x_test_supp <- x_train_supp[part[[2]],]
y_test <- y_train[part[[2]]]


x_train <- x_train[part[[3]],,]
x_train_supp <- x_train_supp[part[[3]],]
y_train <- y_train[part[[3]]]


ts_input <- layer_input(c(dim(x_train)[2], dim(x_train)[3]), name = 'ts_in') 

ts_lstm <- ts_input %>%
  bidirectional(layer_lstm(units = 128, name = 'lstm_1', return_sequences=TRUE, recurrent_regularizer = regularizer_l2())) %>%
  layer_lstm(units=64, name = 'lstm_2', recurrent_regularizer = regularizer_l2())

supp_input <- layer_input(shape = ncol(x_train_supp), name = 'supp_in')

supp_layers <- supp_input %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(32, activation='relu')

concat <- layer_concatenate(list(ts_lstm, supp_layers), name = 'concat')

out_layers <- concat %>%
  layer_dense(units=128, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 64, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 128, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 64, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units=32, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units=32, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 4, activation = "linear") %>%
  layer_distribution_lambda(function(x) {
    tfd_sinh_arcsinh(loc = x[, 1, drop = FALSE],
                     scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]),
                     skewness=x[, 3, drop=FALSE],
                     tailweight= 1e-3 + tf$math$softplus(x[, 4, drop = FALSE]))
  }
  )

model_supp <- keras_model(
  inputs = list(ts_in=ts_input, supp_in=supp_input),
  outputs = out_layers
)


negloglik <- function(y, model) - (model %>% tfd_log_prob(y))

learning_rate <- 0.001

model_supp %>% compile(optimizer = optimizer_adam(lr = learning_rate), loss = negloglik)

history <- model_supp %>% fit(x=list(ts_in=x_train, supp_in=x_train_supp), 
                              y=list(y_train),
                              shuffle=TRUE,
                              validation_data = list(list(ts_in=x_valid, supp_in=x_valid_supp), y_valid),
                              epochs = 200, 
                              batch_size=8000, 
                              callbacks=list(callback_early_stopping(monitor='val_loss', patience = 20),
                                             callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                                                           patience = 10, verbose = 0, mode = 'auto',
                                                                           min_delta = 1e-04, cooldown = 0, min_lr = 0)),
)


#load_model_weights_tf(model_supp, 'stonk_weights_v2.tf')


save_model_weights_tf(model_supp, 'stonk_weights_v2.tf', overwrite = TRUE)


rows_of_interest <- sample(part[[3]], 10000) %>% order()


pred_dist <- model_supp(list(tf$constant(x_test[rows_of_interest,,]), tf$constant(x_test_supp[rows_of_interest,])))


## Evalutate 

out_df <- tibble(
  
  loc = pred_dist$loc %>% as.numeric(),
  scale = pred_dist$scale %>% as.numeric(),
  skewness = pred_dist$skewness %>% as.numeric(),
  tailweight = pred_dist$tailweight %>% as.numeric(),
  
  
  close_last = y_train_df$close_last[part[[3]]][rows_of_interest],
  actual_dlr =y_train_df$close_out1[part[[3]]][rows_of_interest],
  actual_pct = actual_dlr/close_last,
  
  quant10_pct = pred_dist$quantile(.1) %>% as.numeric(), #%>% exp(),
  quant25_pct = pred_dist$quantile(.25) %>% as.numeric(),# %>% exp(),
  quant50_pct = pred_dist$quantile(.5) %>% as.numeric(),# %>% exp(),
  quant75_pct = pred_dist$quantile(.75) %>% as.numeric(),# %>% exp(),
  quant90_pct = pred_dist$quantile(.9) %>% as.numeric(),# %>% exp()
  
  quant10_dlr = quant10_pct*close_last, 
  quant25_dlr = quant25_pct*close_last, 
  quant50_dlr = quant50_pct*close_last, 
  quant75_dlr = quant75_pct*close_last, 
  quant90_dlr = quant90_pct*close_last, 
  
  prob_better_down10 = 1-pred_dist$cdf(.9) %>% as.numeric(),
  prob_better_down05 = 1-pred_dist$cdf(.95) %>% as.numeric(),
  prob_better_down01 = 1-pred_dist$cdf(.99) %>% as.numeric(),
  prob_better_flat = 1-pred_dist$cdf(1) %>% as.numeric(),
  prob_better_up01 = 1-pred_dist$cdf(1.01) %>% as.numeric(),
  prob_better_up05 = 1-pred_dist$cdf(1.05) %>% as.numeric(),
  prob_better_up10 = 1-pred_dist$cdf(1.1) %>% as.numeric(),
  
  act_better_down10 = actual_pct>=.9,
  act_better_down05 = actual_pct>=.95,
  act_better_down01 = actual_pct>=.99,
  act_better_flat = actual_pct>=1,
  act_better_up01 = actual_pct>=1.01,
  act_better_up05 = actual_pct>=1.05,
  act_better_up10 = actual_pct>=1.1
  
  
) 

View(out_df)


## Calibration of Quantiles
out_df %>%
  select(up_10:up_90) %>%
  summarize_each(funs = mean) %>%
  pivot_longer(up_10:up_90, names_to = 'over_quantile', values_to = 'pct') %>%
  ggplot(aes(x=over_quantile, y=pct, label=scales::percent(pct, accuracy = .01))) +
  geom_col() +
  geom_text(vjust=1, color='white') +
  theme_minimal() +
  scale_y_continuous(labels=scales::percent) +
  ggtitle('% Stocks Going Over Specified Quantile')


## Calibration of pred benchmarks
out_df %>%
  mutate(idx=1:nrow(.)) %>%
  select(idx, prob_better_down01:prob_better_up01, act_better_down01:act_better_up01) %>%
  pivot_longer(cols=prob_better_down01:act_better_up01,names_to = c('prob_act', 'direction'), values_to = 'values',
               names_sep = '_better_') %>%
  pivot_wider(values_from = values, names_from = prob_act) %>%
  group_by(direction) %>%
  mutate(prob_cut=cut_number(prob, 5)) %>%
  ungroup() %>%
  arrange(prob) %>%
  mutate(prob_cut = fct_inorder(prob_cut)) %>%
  group_by(direction, prob_cut) %>%
  summarize(act=mean(act),
            n=n()) %>%
  ungroup() %>%
  ggplot(aes(x=prob_cut, y=act, fill=n)) +
  geom_col() +
  facet_wrap(~direction, scales = 'free',
             ncol = 1) +
  coord_flip()



## 
out_df %>%
  ggplot(aes(x=quant50_pct, y=actual_pct)) +
  geom_point(alpha=.3) +
  geom_smooth() +
  coord_cartesian(xlim = c(.99, 1.01))




out_df %>%
  ggplot(aes(x=prob_better_up01, y=actual_pct)) +
  geom_point(alpha=.3) +
  geom_smooth() +
  coord_cartesian(xlim = c(.05, .4))





## Range vs Actual Outcome

out_df %>%
  mutate(range_80 = quant90_pct-quant10_pct,
         range_50 = quant75_pct-quant25_pct) %>%
  select(actual_pct, range_80, range_50) %>%
  pivot_longer(range_80:range_50, names_to = 'range_type', values_to = 'range_val') %>%
  ggplot(aes(x=range_val, y=actual_pct, color=range_type, fill=range_type)) +
  geom_point(alpha=.1) +
  geom_smooth() +
  facet_wrap(~range_type) +
  coord_cartesian(xlim = c(0, .1))



out_df %>%
  mutate(range_80 = quant90_pct-quant10_pct,
         range_50 = quant75_pct-quant25_pct) %>%
  select(actual_pct, range_80, range_50) %>%
  pivot_longer(range_80:range_50, names_to = 'range_type', values_to = 'range_val') %>%
  group_by(range_type) %>%
  mutate(range_cut=cut_number(range_val, 5)) %>%
  group_by(range_type, range_cut) %>%
  summarize(act_sd=sd(actual_pct, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(act_sd) %>%
  mutate(range_cut = fct_inorder(range_cut)) %>%
  ggplot(aes(x=range_cut, y=act_sd)) +
  facet_wrap(~range_type, scales = 'free',
             ncol = 1) +
  geom_col() +
  coord_flip()
  
  

  
  


cor(out_df$actual_pct, out_df$quant50)

out_df %>%
  ggplot(aes(x=quant50_pct, y=actual_pct)) +
  geom_point()

