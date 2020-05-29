library(tidyverse)
library(rsample)
library(riingo)
library(furrr)
library(tictoc)
library(lubridate)
library(reticulate)
library(tensorflow)
library(keras)
library(tfprobability)
library(recipes)
library(AlpacaforR)
plan(multiprocess)

supported <- supported_tickers()

assets <- get_assets()

account <- get_account()

buying_power  <- account$daytrading_buying_power

shortable <- filter(assets, shortable==TRUE)

#use_condaenv('tf2gpu', required = TRUE)

out_df <- readRDS('pred_df_daily.RDS')  %>%
    filter(quant50_pct>.7,
           quant50_pct<1.3,
           volume_last>200000)


high_upside <- out_df %>%
    filter(abs(1-quant75_pct)>abs(1-quant25_pct),
           abs(1-quant90_pct)>abs(1-quant10_pct),
           #skewness>0,
           #loc>1
           ) %>%
    arrange(-prob_better_up01) %>%
    #filter(!ticker %in% 'SRTY') %>%
    slice(1:20)

high_upside <- riingo_iex_quote(high_upside$ticker) %>%
    select(ticker, last, bidPrice, askPrice) %>%
    inner_join(high_upside) %>%
    rowwise() %>%
    mutate(tgt_price = min(quant50_dlr, last, na.rm=TRUE))
    

riingo_meta(high_upside$ticker) %>%
    select(ticker, name)


## Short Positions #######################
low_upside <- out_df %>%
    inner_join(supported %>% filter(assetType=='Stock') %>% select(ticker)) %>% ## Filter to Only Stocks
    filter(ticker %in% shortable$symbol,
           abs(1-quant75_pct)<abs(1-quant25_pct),
           abs(1-quant90_pct)<abs(1-quant10_pct)) %>%
    arrange(prob_better_flat) %>%
    filter(!ticker %in% c('REK')) %>%
    slice(1:20)


low_upside <- riingo_iex_quote(low_upside$ticker) %>%
    select(ticker, last, bidPrice, askPrice) %>%
    inner_join(low_upside) %>%
    rowwise() %>%
    mutate(tgt_price = max(quant50_dlr, last, na.rm = TRUE))


riingo_meta(low_upside$ticker) %>%
    select(ticker, name)




## Enter Long Orders ##################################
#tgt_price <- high_upside$quant50_dlr
    
submit_order_multiple <- function(ticker, last_price, desired_dollar, ...){
    submit_order(ticker = ticker,
                 qty = as.character(round(desired_dollar/last_price)),
                 side = 'buy',
                 type = 'limit',
                 time_in_force = 'day',
                 limit_price = last_price*1.01,
                 extended_hours = TRUE
    )
    
}

ls_inp <- list(ticker = high_upside$ticker,
               last_price = high_upside$tgt_price,
               desired_dollar = (as.numeric(buying_power)/nrow(high_upside))*.4)

long_orders <- pmap(ls_inp, submit_order_multiple)



## Enter SHort Orders #######################################

short_order_multiple <- function(ticker, last_price, desired_dollar, ...){
    submit_order(ticker = ticker,
                 qty = as.character(round(desired_dollar/last_price)),
                 side = 'sell',
                 type = 'limit',
                 time_in_force = 'day',
                 limit_price = last_price,
                 extended_hours = TRUE
    )
    
}


#riingo_iex_quote(low_upside$ticker)

#tgt_short_price <- max(low_upside$quant50_dlr, riingo_iex_quote(low_upside$ticker)$last, na.rm = TRUE)



short_inp <- list(ticker = low_upside$ticker,
                  last_price = low_upside$tgt_price,
                  desired_dollar = (as.numeric(buying_power)/nrow(low_upside))*.4)


short_orders <- pmap(short_inp, short_order_multiple)
