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

stocks = ['WIKI/IBM', 'WIKI/CSCO', 'WIKI/FB', 'WIKI/GOOGL', 'WIKI/MSFT', 'WIKI/AMZN', 'WIKI/TWTR']

def apply(table):
    return lambda a: np.array([x if i == 0 else x/table[0][0][i] - 1 for i, x in enumerate(a[0])])

# TODO: non overlap instead?
def gen_data(indices, length, num_iter):
    np.random.shuffle(stocks)
    # for i in range(num_iter):
    stock_name = stocks[0] # stocks[i % len(stocks)]
    table = all_fields(indices, stock_name).reshape(-1, 1)
    table = np.apply_along_axis(apply(table), axis=1, arr=table)
    # inds = np.arange(0, table.shape[0] - length, length)
    inds = np.arange(0, table.shape[0] - length + 1, length)
    m = max(inds)
    inds = np.array(list(filter(lambda n: n != m, inds)))
    for _ in range(num_iter):
        np.random.shuffle(inds)
        for ind in inds:
            # yield table[ind: ind+length], table[ind+1: ind+length+1]
            yield table[ind: ind+length], table[ind+length: ind+2*length]

# TODO: add transforms?

# test
if __name__ == '__main__':
    for x, y in gen_data([1, 2], 5, 5):
        print(x.shape)
        print('x: {}'.format(x))
        print('y: {}'.format(y))
        break
    plot([1], stocks[0])