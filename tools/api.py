import quandl
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt
quandl.ApiConfig.api_key = "_751heoQxZH7YciZMunN"

tickers = ['GOOG']
data = quandl.get_table('WIKI/PRICES', ticker=tickers, qopts={'columns': ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']}, paginate=True)

data.plot(x='date', y='close')

plt.show()