# Portfolio Optimizer

A portfolio optimization tool that takes any number of stock tickers, simulates 10,000 random weight combinations, plots the efficient frontier, and uses scipy to find the exact portfolio weights that maximize the Sharpe ratio.

<img width="900" height="459" alt="Screenshot 2026-08-10 at 11 19 21 PM" src="https://github.com/user-attachments/assets/262fa054-5744-4f46-bcd2-ab9419c04a8f" />

## What it does

- Takes any number of tickers as user input, one by one
- Downloads 5 years of historical price data for each stock
- Calculates annualized expected returns and a covariance matrix from daily returns
- Simulates 10,000 random portfolios with different weight combinations
- Plots every portfolio as a dot colored by Sharpe ratio using the viridis colormap
- Uses scipy minimize to find the mathematically exact optimal portfolio weights
- Marks the optimal portfolio as a red star on the frontier
- Prints the optimal allocation per ticker

## Why the covariance matrix matters

Combining stocks doesn't just average out their individual volatilities. The actual portfolio volatility depends on how the stocks move relative to each other. Two stocks that always move together give you no diversification benefit. Two stocks that move opposite each other reduce your combined risk below what either one achieves alone.

The covariance matrix captures every pairwise relationship between stocks simultaneously. The matrix multiplication weights.T @ cov_matrix @ weights computes the true collective volatility of any specific weight combination in one operation, accounting for every correlation at once.

This is why portfolio optimization is not just picking the highest returning stocks. It is finding the weight combination that gets the most return per unit of risk, after accounting for how all the stocks interact.

## The Sharpe ratio

Return divided by volatility, minus the risk free rate. Higher is better. It measures how much return you are getting per unit of risk taken. The optimizer minimizes negative Sharpe ratio which is mathematically equivalent to maximizing it.

## The efficient frontier

Each dot is one simulated portfolio. The curved left edge of the cloud is the efficient frontier, the boundary of what is mathematically achievable. No portfolio can exist to the left of that curve because diversification has a mathematical limit. The red star sits on the curve at the point of maximum Sharpe ratio.

## Stack

```
yfinance        — historical price data
pandas          — returns and covariance matrix
numpy           — random weight generation, matrix operations
scipy.optimize  — Sharpe ratio maximization
matplotlib      — efficient frontier scatter plot with colormap
```

## Usage

```bash
pip install yfinance pandas numpy scipy matplotlib
python port_optimizer.py
```

Enter tickers one by one when prompted. Type done when finished.

## Example output

```
Optimal Portfolio Weights:
AAPL:   36.5%
GOOGL:  51.1%
TSLA:    0.0%
GM:     12.4%

Maximum Sharpe Ratio: 0.70
```

A 0% allocation means the optimizer found that stock adds no value to the portfolio at its current risk and return profile relative to the other options.

---

Built to understand Modern Portfolio Theory from the math up, not just run someone else's implementation.
