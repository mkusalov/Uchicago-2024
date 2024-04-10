import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize  # Make sure to include this import
import os
#import model as model


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
        
        self.running_price_paths = train_data.copy()
        
        self.train_data = train_data.copy()
        
        #Calculate expected returns and variance of returns
        self.returns_df = data.pct_change().dropna()
        self.COV_matrix = self.returns_df.cov()
        self.weighty = np.array([0., 0.20871274, 0.24323326, 0.00176664, 0.18435676, 0.])
        
        
        # # Variance of portfolio weights
        # self.var_weights = np.var(self.running_price_paths)
        
        # # Variance of portfolio returns
        # self.var_returns = np.var(self.running_price_paths.sum(axis = 1))
        
        # # Variance of portfolio volatility
        # self.var_volatility = np.var(self.running_price_paths.std(axis = 1))
        
        # # Variance of portfolio sharpe ratio
        # self.var_sharpe_ratio = self.var_sharpe_ratios / self.var_weights
        
        # # Variance of portfolio sortino ratio
        # self.var_sortino_ratio = self.var_returns / self.var_volatility
        
        # Variance of portfolio beta
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
    # def minimum_variance_portfolio(daily_returns):
    #     # Compute the covariance matrix
    #     cov_matrix = np.cov(daily_returns, rowvar=False)
        
    #     # Calculate the inverse of the covariance matrix
    #     inv_cov_matrix = np.linalg.inv(cov_matrix)
        
    #     # Create a vector of ones
    #     ones_vector = np.ones(len(cov_matrix))
        
    #     # Calculate the weights of the minimum variance portfolio
    #     weights = inv_cov_matrix.dot(ones_vector) / ones_vector.dot(inv_cov_matrix).dot(ones_vector)
        
    #     return weights

    def window_COV(self, window_size):
        self.recent_df = self.returns_df.tail(window_size)
        self.COV_matrix = self.recent_df.cov()

    def rebalance(self, weights):
        for i in range(len(weights)):
            if weights[i] < 0:
                weights[i] = 0
        weights = weights/np.sum(weights)
        return weights

    def ewma_cov_pairwise_pd(self, x, y, alpha=0.06):
        x = x.mask(y.isnull(), np.nan)
        y = y.mask(x.isnull(), np.nan)
        covariation = ((x - x.mean()) * (y - y.mean()).dropna())
        return covariation.ewm(alpha=0.06).mean().iloc[-1]

    def ewma_cov_pd(self, rets, alpha=0.06):
        assets = rets.columns
        n = len(assets)
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                cov[i, j] = cov[j, i] = self.ewma_cov_pairwise_pd(
                    rets.iloc[:, i], rets.iloc[:, j], alpha=alpha)
        return pd.DataFrame(cov, columns=assets, index=assets)

    def allocate_portfolio(self, asset_prices):
        one = np.array([1,1,1,1,1,1])
        data.loc[len(data.index)+1] = asset_prices
        self.returns_df = data.pct_change().dropna()
        
        self.window_COV(130)
        global_minimum_weights1 = np.matmul(np.linalg.pinv(self.COV_matrix),one)/np.matmul(one.T,np.matmul(np.linalg.pinv(self.COV_matrix),one)).item(0)
        weighty1 = global_minimum_weights1

        self.COV_matrix = self.ewma_cov_pd(self.returns_df,alpha=0.06)
        global_minimum_weights2 = np.matmul(np.linalg.pinv(self.COV_matrix),one)/np.matmul(one.T,np.matmul(np.linalg.pinv(self.COV_matrix),one)).item(0)
        weighty2 = global_minimum_weights2

        weight_comb = self.rebalance(((.75)*weighty1 + (.25)*weighty2))
        self.weighty = weight_comb
        return weight_comb

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
