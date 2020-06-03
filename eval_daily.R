library(tidyverse)
library(AlpacaforR)
library(riingo)
library(lubridate)
library(furrr)
library(tensorflow)
library(tfprobability)
plan(multiprocess)

## Read Daily File
trading_day <- Sys.Date()
  

out_df <- read_csv( paste0('D:/daily_stock_scoring/pred_dist_', str_replace_all(trading_day, '-', '_'), '.csv'))
  


est_pct_move <- function(loc, scale, skewness, tailweight, close_last, pct){
  1 - as.numeric(tfd_cdf(
    distribution = tfd_sinh_arcsinh(loc=loc, scale=scale, skewness=skewness, tailweight=tailweight),
    value = log(close_last*pct)
  ))
}


today_price <- future_map(unique(out_df$ticker), safely(function(x) riingo_prices(x, start_date = trading_day, end_date = trading_day), otherwise = NA))

non_error_idx <- map_dbl(seq(1:length(today_price)), function(x) is.null(today_price[[x]]$error)) %>% as.logical()

today_price <- map_dfr(seq(1:length(today_price))[non_error_idx], function(x) today_price[[x]]$result)



out_df <- out_df %>%
  left_join(select(today_price, ticker, adjClose)) %>%
  rename(actual_dlr = adjClose) %>%
  mutate(actual_pct = actual_dlr/close_last,
         actual_cdf = est_pct_move(loc, scale, skewness, tailweight, actual_dlr, 1),
         up10_dlr = actual_dlr>quant10_dlr,
         up25_dlr = actual_dlr>quant25_dlr,
         up50_dlr = actual_dlr>quant50_dlr,
         up75_dlr = actual_dlr>quant75_dlr,
         up90_dlr = actual_dlr>quant90_dlr,
         act_better_down10 = actual_pct>=.9,
         act_better_down05 = actual_pct>=.95,
         act_better_down01 = actual_pct>=.99,
         act_better_flat = actual_pct>=1,
         act_better_up01 = actual_pct>=1.01,
         act_better_up05 = actual_pct>=1.05,
         act_better_up10 = actual_pct>=1.1) %>%
  na.omit()

benchmark_df <- tibble(
  over_quantile = c('up10_dlr', 'up25_dlr', 'up50_dlr', 'up75_dlr', 'up90_dlr'),
  pct = c(.9, .75, .5, .25, .1)
)

## Calibration of Quantiles - Add Benchmarks
out_df %>%
  select(up10_dlr:up90_dlr) %>% 
  summarize_each(funs = mean) %>%
  pivot_longer(up10_dlr:up90_dlr, names_to = 'over_quantile', values_to = 'pct') %>%
  ggplot(aes(x=over_quantile, y=pct, label=scales::percent(pct, accuracy = .01))) +
  geom_col(fill='darkgreen') +
  geom_col(data = benchmark_df, 
           mapping = aes(x=over_quantile, y=pct),
           fill=NA, color='black') +
  geom_text(vjust=1, color='white') +
  theme_minimal() +
  scale_y_continuous(labels=scales::percent) +
  ggtitle('% Stocks Going Over Specified Quantile') +
  theme_dark()


## Range vs Actual 
out_df %>%
  mutate(range_80 = quant90_pct-quant10_pct,
         range_50 = quant75_pct-quant25_pct) %>%
  select(actual_pct, range_80, range_50) %>%
  pivot_longer(range_80:range_50, names_to = 'range_type', values_to = 'range_val') %>%
  ggplot(aes(x=range_val, y=actual_pct, color=range_type, fill=range_type)) +
  geom_point(alpha=.1) +
  geom_smooth() +
  facet_wrap(~range_type, scales = 'free') +
  coord_cartesian(xlim = c(0, .1)) +
  theme_dark()


## StandardDev by Range
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
  ggplot(aes(x=range_cut, y=act_sd, label=round(act_sd, 4))) +
  facet_wrap(~range_type, scales = 'free',
             ncol = 1) +
  geom_col(fill = 'darkgreen') +
  geom_text(hjust=1, color='white') +
  coord_flip() +
  theme_dark()


## Calibration of Percentiles
out_df %>%
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
  geom_col(fill = 'darkgreen') +
  geom_text(hjust=1, color='white') +
  facet_wrap(~direction, scales = 'free_y',
             ncol = 1) +
  coord_flip() +
  theme_dark()



## Association of quant50 and actual pct return
out_df %>%
  ggplot(aes(x=quant50_pct, y=actual_pct)) +
  geom_point(alpha=.3, color='green') +
  geom_smooth() +
  coord_cartesian(xlim = c(.95, 1.05)) +
  theme_dark()


## Actual CDF Distribution
out_df %>%
  ggplot(aes(x=actual_cdf)) +
  geom_density(fill = 'darkgreen') +
  geom_density(data = enframe(as.numeric(tfd_sample(tfd_uniform(), 10000))), 
               mapping = aes(x=value), 
               color='black', linetype='dashed', fill='white', alpha=.3) +
  theme_dark()

