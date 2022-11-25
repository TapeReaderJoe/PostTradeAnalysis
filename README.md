# PostTradeAnalysis

## Install dependencies
```
pip install -r requirements.txt
```
## Configuration
Put your transaction csv from avanza in the root folder with the name "transactions.csv" or change the path in
post_trade_analysis.py:164 (the path argument in "read_post_trade_file"). 

## Misc
For now the script supports the Nordic market (SEK, NOK, DKK, FI). 
The ISIN from the transaction csv is mapped to a yahoo ticker from the instruments_meta_data.csv to be able to fetch stock data through the yfinance lib.
