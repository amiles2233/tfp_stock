library(tidyverse)
library(AlpacaforR)


orders <- get_orders(live=FALSE) %>%
  filter(str_detect(order_type, 'stop'))


positions <- get_positions(live=FALSE) %>%
  select(symbol, current_price)



bad_stop <- orders %>%
  inner_join(positions) %>%
  filter((side=='sell' & limit_price>current_price) |
           (side=='buy' & limit_price<current_price)) %>%
  arrange(symbol) %>%
  mutate(sell_price = ifelse(side=='sell', positions$current_price*.995, current_price*1.005))


paste0(nrow(bad_stop), ' Orders Blown Through Stops')

## Cancel the orders

walk(bad_stop$id, cancel_order)


## Sell Off Limit
sell_off_limit <- function(ticker, qty, last_price, side, ...) {
  submit_order(ticker = ticker,
               qty =qty,
               side = side,
               type = 'limit',
               time_in_force = 'day',
               limit_price = last_price,
               extended_hours = TRUE)
}

sell_off_limit_inp <- list(ticker=bad_stop$symbol,
                           qty=abs(bad_stop$qty),
                           side=bad_stop$side,
                           last_price=ifelse(bad_stop$side=='sell', bad_stop$current_price*.995, bad_stop$current_price*1.005))


pmap(.l=sell_off_limit_inp, .f=sell_off_limit)
