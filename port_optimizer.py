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
plt.style.use('dark_background')
stock_list = []

while True:
        stock = str(input("What stock do you want to analyze? (X to exit)"))
        stock_list.append(stock.upper())
        if(stock.lower() == 'x'):
            break
        data[f"Stock{num_stock}"] = pd.DataFrame(yf.download(stock, period='5y'))["Close"].pct_change()
        if(data[f"Stock{num_stock}"].empty):
             print("Try again, please use the ticker symbol!")
             num_stock -= 1
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

    

fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios': [2,5]}, figsize = (10,4.8)) 
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].set_xticks([])
ax[0].set_yticks([])

statement = f"Sharpe Ratio: {-max_sharpe['fun']:.2f}\nStock Weights:"
for i in range(len(max_sharpe['x'])):
     statement += f"\n{stock_list[i]}: {max_sharpe['x'][i]:.1%}"

ax[0].text(0.75,0.8,statement, fontweight = 'bold', fontsize = 14.5, va = "top", ha = 'right')


plt.scatter(vols, returns, c = sharpe_ratios, cmap = "viridis")
plt.scatter(max_sharpe_x, max_sharpe_y, marker = "*", color = "r")
plt.colorbar(label = "Sharpe Ratio")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier")
# plt.subplots_adjust(wspace=0)
plt.show()


    





