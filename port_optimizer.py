import yfinance as yf
from scipy.optimize import minimize 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import math as math 

#1 need returns (pct change over a year)
#2 need portfolio volatility from covariance matrix
# plot random values
# calcualte max sharpe ratio 
#show weights for each stock for the max sharpe 


data = pd.DataFrame()
num_stock = 0

while True:
        stock = str(input("What stock do you want to analyze? (X to exit"))
        if(stock.lower() == 'x'):
            break
        data[f"Stock{num_stock}"] = pd.DataFrame(yf.download(stock, period='1y'))["Close"].pct_change()
        if(data[f"Stock{num_stock}"].empty):
             print("Try again, please use the ticker symbol!")
             continue 
        num_stock += 1


# rng = np.random.default_rng()
# data = pd.DataFrame(yf.download('AAPL', period = '5y'))['Close'].pct_change()
# data["GOOGL"] = pd.DataFrame(yf.download('GOOGL', period = '5y'))['Close'].pct_change()
# data["TSLA"] = pd.DataFrame(yf.download('TSLA', period = '5y'))['Close'].pct_change()
# data["GM"] = pd.DataFrame(yf.download('GM', period = '5y'))['Close'].pct_change()

cov_annual = data.cov()*252
# stock_returns = np.array([data["AAPL"].mean() * 252, data["GOOGL"].mean() * 252, data["TSLA"].mean() * 252, data["GM"].mean() * 252])
stock_returns = []


for i in range(num_stock):
     stock_returns.append(data[f"Stock{i}"].mean() * 252)

stock_returns = np.array(stock_returns)

sharpe_ratios = []



vols = [] # risk 
returns = [] # return
sharpe_ratios = [] # sharpe ratio
max_sharpe_x = 0
max_sharpe_y = 0


def f(x):
    sharpe = (((stock_returns * x).sum() - 0.04) / np.sqrt(cov_annual @ x @ x))
    return -sharpe

    

bnds = []
for i in range(num_stock):
     bnds.append((0,1))

guess = []
for i in range(num_stock):
     guess.append((0.25))



max_sharpe = minimize(f,guess, bounds = bnds, constraints= {'type': 'eq', 'fun': lambda x: x.sum() - 1}, method = "SLSQP")
max_sharpe_x = (np.sqrt(cov_annual @ max_sharpe['x'] @ max_sharpe['x']))
max_sharpe_y = ((stock_returns * max_sharpe['x']).sum())


for i in range(10000):
    num = (np.random.uniform(low = 0, high = 1, size = num_stock))
    sum = num.sum()
    num = num/sum
    returns.append((stock_returns * num).sum())
    vols.append(math.sqrt((cov_annual @ num @ num)))
    sharpe_ratios.append((returns[i] - 0.04)/vols[i])

    


plt.scatter(vols, returns, c = sharpe_ratios, cmap = "viridis")
plt.scatter(max_sharpe_x, max_sharpe_y, marker = "*", color = "r")
plt.colorbar(label = "Sharpe Ratio")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.show()


    





