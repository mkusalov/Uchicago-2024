import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


data = pd.read_csv('Case 2 Data 2024.csv', index_col = 0)

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)


class Allocator():

    def __init__(self, train_data, risk_free_rate=0.02):
        self.train_data = train_data
        self.risk_free_rate = risk_free_rate
        self.running_price_paths = pd.DataFrame()

    def returns_from_prices(self, prices):
        """Calculate daily returns from prices."""
        return prices.pct_change().dropna()

    def calculate_ema_returns(self, returns, span=500, frequency=252):
        """Calculate Exponential Moving Average (EMA) of returns for each asset."""
        # Calculate EMA of daily returns
        ema_returns = returns.ewm(span=span).mean()
        # Assuming returns are daily, convert to annualized returns
        annualized_returns = ema_returns.mean() * frequency
        # Ensure it returns a pandas Series
        annualized_returns = pd.Series(annualized_returns, index=returns.columns)
        return annualized_returns


    def calculate_ewm_covariance(self, returns, span=500, frequency=252):
        """Calculate Exponential Weighted Moving (EWM) Covariance of returns."""
        ewm_cov = returns.ewm(span=span).cov().groupby(level=0).last() * frequency
        return ewm_cov

    def neg_sharpe_ratio(self, weights, mu, sigma):
        """Negative Sharpe Ratio for optimization."""
        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return - (portfolio_return - self.risk_free_rate) / portfolio_volatility

    def maximize_sharpe_ratio(self, mu, sigma):
        """Optimize portfolio to maximize Sharpe Ratio."""
        num_assets = len(mu)
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        initial_guess = [1. / num_assets] * num_assets
        
        result = minimize(self.neg_sharpe_ratio, initial_guess, args=(mu, sigma),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def allocate_portfolio(self, current_prices):
        """Determine portfolio allocation for given prices."""
        if self.running_price_paths.empty:
            self.running_price_paths = current_prices
        else:
            self.running_price_paths = self.running_price_paths.append(current_prices, ignore_index=True)
        
        returns = self.returns_from_prices(self.running_price_paths)
        mu = self.calculate_ema_returns(returns)
        sigma = self.calculate_ewm_covariance(returns)
        
        optimal_weights = self.maximize_sharpe_ratio(mu, sigma)
        return optimal_weights

    #new

    # def __init__(self, train_data, risk_free_rate=0.02):
    #     self.running_price_paths = train_data
    #     self.risk_free_rate = risk_free_rate

    # def returns_from_prices(self, prices):
    #     return prices.pct_change().dropna()

    # def calculate_ema_returns(self, prices, span=500, frequency=252):
    #     daily_returns = self.returns_from_prices(prices)
    #     ema_returns = daily_returns.ewm(span=span).mean().iloc[-1]
    #     annualized_returns = (1 + ema_returns) ** frequency - 1
    #     return annualized_returns

    # def calculate_ewm_covariance(self, returns, span=500):
    #     return returns.ewm(span=span).cov().iloc[-len(returns.columns):]

    # def neg_sharpe_ratio(self, weights, mu, sigma):
    #     Rp = np.dot(weights, mu)
    #     volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
    #     return -(Rp - self.risk_free_rate) / volatility

    # def maximize_sharpe_ratio(self, mu, sigma):
    #     num_assets = len(mu)
    #     args = (mu, sigma)
    #     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    #     bounds = [(0, 1) for _ in range(num_assets)]
    #     initial_guess = [1. / num_assets] * num_assets
    #     result = minimize(self.neg_sharpe_ratio, initial_guess, args=args,
    #                       method='SLSQP', bounds=bounds, constraints=constraints)
    #     return result.x

    # def allocate_portfolio(self, current_prices):
    #     self.running_price_paths = pd.concat([self.running_price_paths, pd.DataFrame([current_prices], columns=self.running_price_paths.columns)], ignore_index=True)
    #     mu = self.calculate_ema_returns(self.running_price_paths)
    #     returns = self.returns_from_prices(self.running_price_paths)
    #     sigma = self.calculate_ewm_covariance(returns)
    #     optimal_weights = self.maximize_sharpe_ratio(mu, sigma)
    #     return optimal_weights

    
    
    #old::
    
    
    # def returns_from_prices(self, prices, log_returns=False):
    #     if log_returns:
    #         returns = np.log(1 + prices.pct_change()).dropna(how="all")
    #     else:
    #         returns = prices.pct_change().dropna(how="all")
    #     return returns

    # def ema_historical_return(self,
    #     prices,
    #     returns_data=False,
    #     compounding=True,
    #     span=500,
    #     frequency=252,
    #     log_returns=False,
    #     ):
    #     if returns_data:
    #         returns = prices
    #     else:
    #         returns = self.returns_from_prices(prices)

    #     return (1 + returns.ewm(span=span).mean().iloc[-1]) ** frequency - 1
    
    # def ewm_covariance_matrix(self, returns, decay_factor):
    #     # Calculate the weights
    #     n = len(returns)
    #     weights = decay_factor ** np.arange(n)[::-1]
    #     weights /= weights.sum()
        
    #     # Apply weights to returns
    #     weighted_returns = returns.mul(weights, axis=0)
        
    #     # Calculate the mean-adjusted returns
    #     weighted_mean_returns = weighted_returns.sum()
    #     mean_adjusted_returns = weighted_returns - weighted_mean_returns
        
    #     # Initialize the covariance matrix
    #     ewm_cov_matrix = np.zeros((len(returns.columns), len(returns.columns)))
        
    #     # Calculate the weighted covariance matrix
    #     for i in range(len(returns.columns)):
    #         for j in range(len(returns.columns)):
    #             # Utilize .iloc for safe row selection and direct column names for clarity
    #             column_i = mean_adjusted_returns.iloc[:, i]
    #             column_j = mean_adjusted_returns.iloc[:, j]
    #             ewm_cov_matrix[i, j] = (column_i * column_j).sum()
    #     # Calculate the weights
    #     # n = returns.shape[0]
    #     # weights = np.array([decay_factor**(n-1-i) for i in range(n)])
    #     # weights /= weights.sum()
        
    #     # # Calculate the mean-adjusted returns
    #     # weighted_means = np.sum(returns * weights[:, np.newaxis], axis=0)
    #     # mean_adjusted_returns = returns - weighted_means
        
    #     # # Calculate the weighted covariance matrix
    #     # ewm_cov_matrix = np.zeros((returns.shape[1], returns.shape[1]))
    #     # for i in range(returns.shape[1]):
    #     #     for j in range(returns.shape[1]):
    #     #         ewm_cov_matrix[i, j] = np.sum(weights * mean_adjusted_returns[:, i] * mean_adjusted_returns[:, j])
        
    #     return ewm_cov_matrix

    # Rf = 0.02  # Example risk-free rate

    # def neg_sharpe_ratio(self, weights, mu, sigma, Rf):
    #     """Return the negative of the Sharpe ratio for the given portfolio weights."""
    #     Rp = np.dot(weights, mu)  # Portfolio return
    #     volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))  # Portfolio volatility
    #     return -(Rp - Rf) / volatility  # Negative Sharpe ratio

    # def maximize_sharpe_ratio(self, mu, sigma, Rf):
    #     """Find the portfolio weights that maximize the Sharpe ratio."""
    #     num_assets = len(mu)
    #     args = (mu, sigma, Rf)
    #     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights = 1
    #     bounds = tuple((0, 1) for asset in range(num_assets))
    #     initial_guess = num_assets * [1. / num_assets, ]
        
    #     result = minimize(self.neg_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        
    #     return result
    # def __init__(self, train_data):
    #     '''
    #     Anything data you want to store between days must be stored in a class field
    #     '''
        
    #     # Convert asset_prices to a DataFrame
    #     self.running_price_paths = train_data.copy()
        
    #     self.train_data = train_data.copy()
        
    #     #Calculate expected returns and variance of returns
    #     asset_prices = self.running_price_paths.pct_change().dropna()

    #     # self.running_price_paths = train_data.copy()
        
    #     # self.train_data = train_data.copy()
        
    #     # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
        
    # def allocate_portfolio(self, asset_prices):
    #     '''
    #     asset_prices: np array of length 6, prices of the 6 assets on a particular day
    #     weights: np array of length 6, portfolio allocation for the next day
    #     '''
    #     # Assuming asset_prices is a numpy array and self.running_price_paths.columns provides the correct column names
    #     asset_prices_df = pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)
    #     self.running_price_paths = pd.concat([self.running_price_paths, asset_prices_df], ignore_index=True)

    
    #     ### TODO Implement your code here
    #     historical_returns = self.returns_from_prices(self.running_price_paths)

    # # Calculate expected returns (mu) using the ema_historical_return method on the historical returns
    #     mu = self.ema_historical_return(historical_returns, returns_data=True)

    # # Calculate the covariance matrix using the historical returns
    #     sigma = self.ewm_covariance_matrix(historical_returns, decay_factor=0.94) 
    #     Rf = 0
    #     optimization_result = self.maximize_sharpe_ratio(mu, sigma, Rf)
    #     weights = optimization_result.x
        
    #     return weights


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