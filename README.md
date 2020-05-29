# Applying Tensorflow Probability to the Stock Market

### Goal

Model the next day's close price using Deep Learning Yearning and Tensorflow Probability

Next steps are to:

* Model the next day's high and low prices

* Make a new model to use open price as a predictor

* Expand horizon to multiple days outs of weeks out either by explicit modeling or chaining

### File Explanation

`yearning_stock_ts.R`: Model training

`score_yearning_model_ts.R`: Model evaluation

`daily_scoring.R`: Pull all tickers from Tiingo, and get a predicted distribution for each ticker on a given day

`daily_order_entry.R`: Use the output from `daily_scoring.R` to make orders with Alpaca

`mid_day_rebalance.R`: Deploy daily mean reversion strategy using the output from `daily_scoring.R` and submit orders with Alpaca

`sell_off.R`: Create a limit order for all currently held stocks with Alpaca

`stop_loss.R`: Submit stop loss orders for positions based on daily scoring.

`stonk_weights*`: Weights for Deep Learning model.

`*transform_recipe.RDS`: Supplementary data transformation using the `recipes` package.
