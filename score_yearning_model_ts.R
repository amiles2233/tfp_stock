library(tidyverse)
library(reticulate)
library(tensorflow)
library(tfprobability)
library(keras)

use_condaenv('tf2gpu', required = TRUE)

### For Aaron's COMP #################################
## Make sure tf gpu doesn;t f up
py_config()

tf_config()

tf_gpu_configured()

md <- keras_model_sequential() %>% layer_dense(units=2, input_shape=5, activation='relu')

rm(md)



x_test <- readRDS('D:/x_test_array.RDS')
x_test_supp <- readRDS('D:/x_test_supp.RDS')
y_test_df <- readRDS('D:/y_test.RDS')

rows_of_interest <- sample(1:nrow(y_test_df), 100000) %>% order()


x_test <- x_test[rows_of_interest,,]
x_test_supp <- x_test_supp[rows_of_interest,]
y_test_df <- y_test_df[rows_of_interest,]


y_test <- log(y_test_df$close_out1)

x_test_supp <- as.matrix(x_test_supp)



ts_input <- layer_input(c(dim(x_test)[2], dim(x_test)[3]), name = 'ts_in') 

ts_lstm <- ts_input %>%
  bidirectional(layer_lstm(units = 128, name = 'lstm_1', return_sequences=TRUE, recurrent_regularizer = regularizer_l2())) %>%
  layer_lstm(units=64, name = 'lstm_2', recurrent_regularizer = regularizer_l2())

supp_input <- layer_input(shape = ncol(x_test_supp), name = 'supp_in')

supp_layers <- supp_input %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(32, activation='relu')

concat <- layer_concatenate(list(ts_lstm, supp_layers), name = 'concat')

out_layers <- concat %>%
  layer_dense(units=128, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 128, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 128, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 64, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 64, activation = 'relu', regularizer_l1_l2()) %>%
  layer_dense(units = 128, activation = 'relu', regularizer_l1_l2()) %>%
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

load_model_weights_tf(model_supp, 'stonk_weights_v2_log.tf')


pred_dist <- model_supp(list(tf$constant(x_test), tf$constant(x_test_supp)))


est_pct_move <- function(loc, scale, skewness, tailweight, close_last, pct){
  1 - as.numeric(tfd_cdf(
    distribution = tfd_sinh_arcsinh(loc=loc, scale=scale, skewness=skewness, tailweight=tailweight),
    value = log(close_last*pct)
  ))
}



out_df <- tibble(

  
  loc = pred_dist$loc %>% as.numeric(),
  scale = pred_dist$scale %>% as.numeric(),
  skewness = pred_dist$skewness %>% as.numeric(),
  tailweight = pred_dist$tailweight %>% as.numeric(),
  
  
  close_last = y_test_df$close_last,
  actual_dlr =y_test_df$close_out1,
  actual_pct = actual_dlr/close_last,
  actual_dlr_cdf = est_pct_move(loc, scale, skewness, tailweight, actual_dlr, 1),
  
  quant10_dlr = pred_dist$quantile(.1) %>% as.numeric() %>% exp(),
  quant25_dlr = pred_dist$quantile(.25) %>% as.numeric() %>% exp(),
  quant50_dlr = pred_dist$quantile(.5) %>% as.numeric() %>% exp(),
  quant75_dlr = pred_dist$quantile(.75) %>% as.numeric() %>% exp(),
  quant90_dlr = pred_dist$quantile(.9) %>% as.numeric() %>% exp(),
  
  quant10_pct = quant10_dlr/close_last, 
  quant25_pct = quant25_dlr/close_last, 
  quant50_pct = quant50_dlr/close_last, 
  quant75_pct = quant75_dlr/close_last, 
  quant90_pct = quant90_dlr/close_last,
  
  up10_dlr = actual_dlr>quant10_dlr,
  up25_dlr = actual_dlr>quant25_dlr,
  up50_dlr = actual_dlr>quant50_dlr,
  up75_dlr = actual_dlr>quant75_dlr,
  up90_dlr = actual_dlr>quant90_dlr,
  
  up10_pct = actual_pct>quant10_pct,
  up25_pct = actual_pct>quant25_pct,
  up50_pct = actual_pct>quant50_pct,
  up75_pct = actual_pct>quant75_pct,
  up90_pct = actual_pct>quant90_pct,
  
  prob_better_down10 = est_pct_move(loc, scale, skewness, tailweight, close_last, .9),
  prob_better_down05 = est_pct_move(loc, scale, skewness, tailweight, close_last, .95),
  prob_better_down01 = est_pct_move(loc, scale, skewness, tailweight, close_last, .99),
  prob_better_flat = est_pct_move(loc, scale, skewness, tailweight, close_last, 1),
  prob_better_up01 = est_pct_move(loc, scale, skewness, tailweight, close_last, 1.01),
  prob_better_up05 = est_pct_move(loc, scale, skewness, tailweight, close_last, 1.05),
  prob_better_up10 = est_pct_move(loc, scale, skewness, tailweight, close_last, 1.1),
  
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
  #sample_n(1000) %>%
  select(up10_dlr:up90_dlr) %>% 
  summarize_each(funs = mean) %>%
  pivot_longer(up10_dlr:up90_dlr, names_to = 'over_quantile', values_to = 'pct') %>%
  ggplot(aes(x=over_quantile, y=pct, label=scales::percent(pct, accuracy = .01))) +
  geom_col() +
  geom_text(vjust=1, color='white') +
  theme_minimal() +
  scale_y_continuous(labels=scales::percent) +
  ggtitle('% Stocks Going Over Specified Quantile')


## Range vs Actual Outcome

out_df %>%
  sample_n(1000) %>%
  mutate(range_80 = quant90_pct-quant10_pct,
         range_50 = quant75_pct-quant25_pct) %>%
  select(actual_pct, range_80, range_50) %>%
  pivot_longer(range_80:range_50, names_to = 'range_type', values_to = 'range_val') %>%
  ggplot(aes(x=range_val, y=actual_pct, color=range_type, fill=range_type)) +
  geom_point(alpha=.1) +
  geom_smooth() +
  facet_wrap(~range_type, scales = 'free') +
  coord_cartesian(xlim = c(0, .1))

## StandardDev by Range
out_df %>%
  #sample_n(1000) %>%
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
  ggplot(aes(x=range_cut, y=act_sd, label=round(act_sd, 4))) +
  facet_wrap(~range_type, scales = 'free',
             ncol = 1) +
  geom_col() +
  geom_text(hjust=1, color='white') +
  coord_flip()




## Calibration of pred benchmarks
out_df %>%
  #sample_n(1000) %>%
  mutate(idx=1:nrow(.)) %>%
  select(idx, prob_better_down01:prob_better_up01, act_better_down01:act_better_up01) %>%
  pivot_longer(cols=prob_better_down01:act_better_up01,names_to = c('prob_act', 'direction'), values_to = 'values',
               names_sep = '_better_') %>%
  pivot_wider(values_from = values, names_from = prob_act) %>%
  group_by(direction) %>%
  mutate(prob_cut=cut_number(prob,5)) %>%
  ungroup() %>%
  arrange(prob) %>%
  mutate(prob_cut = fct_inorder(prob_cut)) %>%
  group_by(direction, prob_cut) %>%
  summarize(act=mean(act),
            n=n()) %>%
  ungroup() %>%
  ggplot(aes(x=prob_cut, y=act, label = scales::percent(act))) +
  geom_col() +
  geom_text(hjust=1, color='white') +
  facet_wrap(~direction, scales = 'free_y',
             ncol = 1) +
  coord_flip()



## Association of Quant50 and Actual PCT
out_df %>%
  sample_n(1000) %>%
  ggplot(aes(x=quant50_pct, y=actual_pct)) +
  geom_point(alpha=.3) +
  geom_smooth() +
  coord_cartesian(xlim = c(.95, 1.05))



cor(out_df$quant50_pct, out_df$actual_pct)

out_df %>%
  sample_n(1000) %>%
  ggplot(aes(x=prob_better_up01, y=actual_pct)) +
  geom_point(alpha=.3) +
  geom_smooth() +
  coord_cartesian(xlim = c(.05, .4))



## CDFs of Actual Closes
out_df %>%
  ggplot(aes(x=actual_dlr_cdf)) +
  geom_histogram()
