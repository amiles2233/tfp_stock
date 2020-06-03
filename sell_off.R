library(tidyverse)
library(AlpacaforR)


## Cancel ALL Orders

cancel_order(all = TRUE)

positions <- get_positions()

sell_off_limit <- function(ticker, qty, last_price, side, ...) {
    submit_order(ticker = ticker,
                 qty =qty,
                 side = side,
                 type = 'limit',
                 time_in_force = 'day',
                 limit_price = last_price,
                 extended_hours = TRUE)
}

sell_off_limit_inp <- list(ticker=positions$symbol,
                           qty=abs(positions$qty),
                           side=ifelse(positions$side=='long', 'sell', 'buy'),
                           last_price=ifelse(positions$side=='long', positions$current_price*.995, positions$current_price*1.005))


pmap(.l=sell_off_limit_inp, .f=sell_off_limit)