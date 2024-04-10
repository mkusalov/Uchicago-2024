import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize  # Make sure to include this import
import os


data = pd.read_csv('Case 2 Data 2024.csv', index_col = 0)

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)


class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        
        self.train_data = train_data.copy()
        self.running_price_paths = train_data.copy()
        self.hist_mean = self.train_data.mean()  # Historical mean returns
        self.hist_cov = self.train_data.cov()  # Historical covariance matrix
        self.bounds = [(-1, 1) for _ in range(len(self.train_data.columns))]  # Assuming short selling is allowed
        
        # You can adjust these parameters
        self.target_returns = np.linspace(0.06, 0.17, 100)
        self.equally_weighted_weights = np.array([1/len(self.train_data.columns)] * len(self.train_data.columns))
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
        
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''
        # self.running_price_paths = self.running_price_paths.append(pd.DataFrame(asset_prices).T, ignore_index=True)
        daily_returns = self.running_price_paths.pct_change().dropna()
        
        # Efficient frontier and minimum variance logic could go here
        # For example, to find minimum variance portfolio for today:
        covariance_matrix = daily_returns.cov()
        #Calculate volatilities
        volatilities = daily_returns.std()
        weights = self.find_minimum_variance_portfolio()
        cov_matrix = daily_returns.cov()
        weights = np.dot(weights, np.dot(cov_matrix, weights))/np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        return weights
    
    def find_minimum_variance_portfolio(self):
        optimal = minimize(
            fun=self.portfolio_std,
            args=(self.hist_cov,),
            x0=self.equally_weighted_weights,
            bounds=self.bounds,
            constraints=(
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            ),
            method='SLSQP'
        )
        return optimal['x']

    def portfolio_std(self, weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    


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

