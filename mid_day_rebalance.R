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

#use_condaenv('tf2gpu', required = TRUE)

acct <- get_account()
positions <- get_positions()
supported <- supported_tickers()


out_df <- readRDS('pred_df_daily.RDS') %>%
  filter(! ticker %in% 'AOBC')

quotes <- future_map(out_df$ticker, safely(riingo_iex_quote, otherwise = NA), .progress = TRUE)
non_error_idx <- map_dbl(seq(1:length(quotes)), function(x) is.null(quotes[[x]]$error)) %>% as.logical()

quotes <- map_dfr(seq(1:length(quotes))[non_error_idx], function(x) quotes[[x]]$result)



quotes <- quotes %>%
  filter(as.Date(quoteTimestamp)==Sys.Date()) %>%
  select(ticker, last, bidPrice, askPrice)



intraday <- out_df %>%
  inner_join(quotes, by='ticker') %>%
  inner_join(supported %>% filter(assetType=='Stock') %>% select(ticker)) %>%
  filter(!ticker %in% positions$symbol) %>%
  mutate(last_prop=last/close_last) 


get_intraday_cdf <- function(value, loc, scale, skewness, tailweight){
  tfd_cdf(distribution = tfd_sinh_arcsinh(loc = loc,
                                          scale = scale,
                                          skewness = skewness,
                                          tailweight = tailweight
  ),
  log(value)) %>%
    as.numeric()
  
}


cdf_inp <- list(value=intraday$last,
                loc=intraday$loc,
                scale=intraday$scale,
                skewness=intraday$skewness,
                tailweight=intraday$tailweight)


intraday$intra_cdf <- 1-future_pmap_dbl(.l=cdf_inp, .f=get_intraday_cdf, .progress = TRUE)


longs <- intraday %>%
  arrange(-intra_cdf) %>%
  slice(1:10) %>%
  select(ticker, intra_cdf, close_last, last_prop, last, bidPrice, askPrice)


long_intraday_multiple <- function(ticker, last_price, desired_dollar, ...){
  submit_order(ticker = ticker,
               qty = as.character(round(desired_dollar/last_price)),
               side = 'buy',
               type = 'limit', ## Figure Out IOC
               time_in_force = 'day',
               limit_price = last_price,
               extended_hours = TRUE
  )
  
}

long_inp <- list(ticker = longs$ticker,
                 last_price = longs$last,
                 desired_dollar = (as.numeric(acct$daytrading_buying_power)/nrow(longs))*.48)

riingo_meta(long_inp$ticker) %>%
  select(ticker, name)




shorts <- intraday %>%
  arrange(intra_cdf) %>%
  slice(1:10) %>%
  select(ticker, close_last, last_prop, last, bidPrice, askPrice) %>%
  filter(!ticker %in% c('DGAZ', 'JDST'))


short_intraday_multiple <- function(ticker, last_price, desired_dollar, ...){
  submit_order(ticker = ticker,
               qty = as.character(round(desired_dollar/last_price)),
               side = 'sell',
               type = 'limit', ## Figure Out IOC
               time_in_force = 'day',
               limit_price = last_price*1.01,
               extended_hours = TRUE
  )
  
}

short_inp <- list(ticker = shorts$ticker,
                  last_price = shorts$last,
                  desired_dollar = (as.numeric(acct$daytrading_buying_power)/nrow(shorts))*.48)

riingo_meta(short_inp$ticker) %>%
  select(ticker, name)


long_intra_orders <- pmap(long_inp, long_intraday_multiple)

short_intra_orders <- pmap(short_inp, short_intraday_multiple)



##
a <- get_meta(high_upside$ticker[3], endpoint = 'news')



get_poly_news <- function(symbol) {
  get_meta(symbol, endpoint = 'news')
}


get_riingo_news <- function(ticker) {
  df <- riingo_news(ticker)
  
  df$query_ticker=ticker
  
  return(df)
}


news_all <- future_map_dfr(high_upside$ticker[15:20], get_riingo_news)




news_all_mod <- news_all %>%
  mutate(today=as.Date(crawlDate)==Sys.Date(),
         yesterday=as.Date(crawlDate)==Sys.Date()-1,
         this_week=as.Date(crawlDate)>=Sys.Date()-7) %>% 
  #unnest(query_ticker) %>%
  group_by(query_ticker) %>% 
  summarize(today=sum(today), yesterday=sum(yesterday), this_week=sum(this_week))

a$endpoints$news

news <- riingo_news(high_upside$ticker[2])
news
