library(tidyverse)
library(AlpacaforR)

out_df <- readRDS('pred_df_daily.RDS')

orders <- get_orders(live=FALSE, status = 'open')

positions <- get_positions(live = FALSE) %>%
  left_join(out_df, by=c('symbol' = 'ticker')) %>%
  filter(! symbol %in% orders$symbol) %>% # Remove tickers with active orders
  distinct() %>%
  group_by(symbol) %>%
  mutate(stop_limit_price = ifelse(side=='long',
                                   min(c(quant10_dlr, current_price*.97)),
                                   max(c(quant90_dlr, current_price*1.07)))) %>%
  ungroup()


ls_inp <- list(ticker = positions$symbol,
               qty = abs(positions$qty),
               side = ifelse(positions$side=='long', 'sell', 'buy'),
               price = positions$stop_limit_price) 

submit_stop_limit = function(ticker, qty, side, price, ...) {
  submit_order(ticker = ticker,
               qty =qty,
               side = side,
               type = 'stop_limit',
               time_in_force = 'day',
               limit_price = price,
               stop_price = price,
               extended_hours = FALSE,
               live = FALSE)
}

stop_limit_orders <- pmap(.l=ls_inp, .f=slowly(submit_stop_limit, rate = rate_delay(1), quiet = TRUE))


length(stop_limit_orders)



rejected <- get_orders(status = 'all', after = Sys.Date()-days(1)) %>%
  filter(status=='rejected')


rejected_assets <- map_dfr(unique(rejected$symbol), get_assets)
