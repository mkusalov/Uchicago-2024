import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


data = pd.read_csv('Case 2 Data 2024.csv', index_col = 0, skiprows = 1)

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.5, shuffle = True)



class Allocator():
    #  new new


    def __init__(self, train_data, risk_free_rate=0.02, return_threshold=0.05, volatility_threshold=0.25):
        self.running_price_paths = train_data
        self.risk_free_rate = risk_free_rate
        self.return_threshold = return_threshold
        self.volatility_threshold = volatility_threshold

    def returns_from_prices(self, prices):
        return prices.pct_change().dropna()

    def calculate_ema_returns(self, prices, span=500, frequency=252):
        daily_returns = self.returns_from_prices(prices)
        ema_returns = daily_returns.ewm(span=span).mean().iloc[-1]
        annualized_returns = (1 + ema_returns) ** frequency - 1
        return annualized_returns

    def calculate_ewm_covariance(self, returns, span=500):
        return returns.ewm(span=span).cov().iloc[-len(returns.columns):]

    def neg_sharpe_ratio(self, weights, mu, sigma):
        Rp = np.dot(weights, mu)
        volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -(Rp - self.risk_free_rate) / volatility

    def identify_riskiest_stocks(self, sigma):
        """
        Identify the two stocks with the highest volatility (risk).
        Returns the names (or indices) of the two riskiest stocks.
        """
        volatilities = np.sqrt(np.diag(sigma))
        riskiest_stocks = pd.Series(volatilities, index=sigma.columns).nlargest(2).index
        return riskiest_stocks

    def calculate_portfolio_volatility(self, weights, sigma):
        """
        Calculate the total volatility of the portfolio.
        """
        portfolio_variance = np.dot(weights.T, np.dot(sigma, weights))
        return np.sqrt(portfolio_variance)

    def calculate_risk_contributions(self, weights, sigma):
        """
        Calculate the contribution of each asset to the portfolio risk.
        """
        portfolio_volatility = self.calculate_portfolio_volatility(weights, sigma)
        marginal_contrib = np.dot(sigma, weights)
        risk_contributions = np.multiply(weights, marginal_contrib) / portfolio_volatility
        return risk_contributions

    def risk_parity_objective(self, weights, args):
        """
        Objective function for risk parity: minimize the squared differences
        between each asset's risk contribution and the target risk contribution.
        """
        sigma = args[0]
        total_risk_contributions = np.sum(self.calculate_risk_contributions(weights, sigma))
        target_contrib = total_risk_contributions / len(weights)
        risk_contributions = self.calculate_risk_contributions(weights, sigma)
        return np.sum((risk_contributions - target_contrib) ** 2)

    def allocate_portfolio_risk_parity(self, current_prices):
        """
        Allocate the portfolio based on risk parity.
        """
        self.running_price_paths = pd.concat([self.running_price_paths, pd.DataFrame([current_prices], columns=self.running_price_paths.columns)], ignore_index=True)
        returns = self.returns_from_prices(self.running_price_paths)
        sigma = self.calculate_ewm_covariance(returns)
        num_assets = len(sigma.columns)
        
        # Initial guess: equal weighting
        initial_weights = np.array([1. / num_assets] * num_assets)
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(self.risk_parity_objective, initial_weights, args=[sigma], method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_allocation = pd.Series(optimal_weights, index=self.running_price_paths.columns)
            return portfolio_allocation
        else:
            raise ValueError("Risk parity optimization failed.")

    def maximize_sharpe_ratio(self, mu, sigma):
        riskiest_stocks = self.identify_riskiest_stocks(sigma)
        num_assets = len(mu)
        args = (mu, sigma)
        
        # Base constraint: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Additional constraints: set weights of riskiest stocks to 0
        for stock in riskiest_stocks:
            stock_idx = mu.index.get_loc(stock)  # Get index of the stock in mu
            constraint = {'type': 'eq', 'fun': lambda x, idx=stock_idx: x[idx]}
            constraints.append(constraint)
        
        bounds = [(0, 1) for _ in range(num_assets)]
        initial_guess = [1. / num_assets] * num_assets
        
        result = minimize(self.neg_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x, mu.index

    def allocate_portfolio(self, current_prices, strategy="sharpe"):
        """
        Allocate the portfolio based on the selected strategy.
        :param current_prices: Current market prices of assets.
        :param strategy: Strategy to use for portfolio allocation. Options are "sharpe" or "risk_parity".
        :return: Portfolio allocation.
        """
        self.running_price_paths = pd.concat([self.running_price_paths, pd.DataFrame([current_prices], columns=self.running_price_paths.columns)], ignore_index=True)
        mu = self.calculate_ema_returns(self.running_price_paths)
        returns = self.returns_from_prices(self.running_price_paths)
        sigma = self.calculate_ewm_covariance(returns)

        if strategy == "sharpe":
            optimal_weights, _ = self.maximize_sharpe_ratio(mu, sigma)
            portfolio_allocation = pd.Series(optimal_weights, index=self.running_price_paths.columns)
        elif strategy == "risk_parity":
            portfolio_allocation = self.allocate_portfolio_risk_parity(current_prices)
        else:
            raise ValueError("Invalid strategy specified.")

        return portfolio_allocation


def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()