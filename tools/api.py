import quandl
import numpy as np
import numpy.lib.recfunctions
import pandas
import matplotlib.pyplot as plt
quandl.ApiConfig.api_key = "_751heoQxZH7YciZMunN"

# table

tickers = ['MSFT']
#data = quandl.get_table('WIKI/PRICES', ticker=tickers, qopts={'columns': ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']}, paginate=True)

#data.plot(x='date', y='close')
#plt.show()
#print(data)

# time-series

cols = {'Date': 0, 'Open': 1, 'High': 2, 'Low': 3, 'Close': 4, 'Volume': 5}

def all_fields(indices, stock_name):
    if len(indices) <= 0:
        return
    arr = quandl.get(stock_name, column_index=indices[0], collapse='daily', returns='numpy')
    for i in range(len(indices)):
        if i != 0:
            ind = indices[i]
            field = quandl.get(stock_name, column_index=ind, collapse='daily', returns='numpy')
            col_name = field.dtype.names[1]
            arr = numpy.lib.recfunctions.append_fields(arr, col_name, field[col_name], usemask=False)
    return arr

def plot(indices, stock_name):
    table = all_fields(indices, stock_name)
    data = pandas.DataFrame.from_records(table)
    data.plot(x='Date', y=list(table.dtype.names)[1:], subplots=True)
    plt.show()

# plot([1, 2, 3], 'WIKI/AAPL')

def gen_data(indices, stock_name, batch_size):
    table = all_fields(indices, stock_name)
    for i in range(table.shape[0] - batch_size):
        yield table[i:i+batch_size], table[i+1:i+batch_size+1]

# test
for x, y in gen_data([1,2], 'WIKI/GOOGL', 5):
    print('x: {}'.format(x))
    print('y: {}'.format(y))
    break