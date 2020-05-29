library(tidyverse)
library(AlpacaforR)

out_df <- readRDS('pred_df_daily.RDS')

orders <- get_orders(live=FALSE, status = 'open')

positions <- get_positions(live = FALSE) %>%
  left_join(out_df, by=c('symbol' = 'ticker')) %>%
  filter(! symbol %in% orders$symbol) %>% # Remove tickers with active orders
  group_by(symbol) %>%
  mutate(stop_limit_price = ifelse(side=='long',
                                   min(c(market_value*.95, quant10_dlr)),
                                   max(c(market_value*1.05, quant90_dlr)))) %>%
  ungroup()


ls_inp <- list(ticker = positions$symbol,
               qty = positions$qty,
               side = ifelse(positions$side=='long', 'sell', 'buy'),
               price = positions$stop_limit_price) 

submit_stop_limit = function(ticker, qty, side, price, ...) {
  submit_order(ticker = ticker,
               qty =qty,
               side = side,
               type = 'stop_limit',
               time_in_force = 'day',
               limit_price = price,
               stop_price = price*.999,
               extended_hours = FALSE,
               live = FALSE)
}

pmap(.l=ls_inp, .f=submit_stop_limit)
