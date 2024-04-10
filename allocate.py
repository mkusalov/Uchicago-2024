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
        daily_returns = self.running_price_paths.pct_change().dropna()
        expected_returns = daily_returns.mean()
        
        # variance of sharpe ratios
        self.var_sharpe_ratios = np.var(daily_returns.mean()) / np.var(expected_returns)
        
        #variance of returns??? TODO: IDK IF THIS IS RIGHT
        self.var_returns = np.var(daily_returns.sum(axis = 1))
        
        #if variance of sharpe ratios bigger than variance of returns, use max diversification
        # else use min variance
        if self.var_sharpe_ratios > self.var_returns:
            self.max_diversification = False
            # execute min variance...
        else:
            #execute maximum diversification on the day to day
            self.max_diversification = True

        
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

    def min_variance_portfolio(self, cov_matrix):
        num_assets = cov_matrix.shape[0]
        initial_guess = np.repeat(1/num_assets, num_assets)
        bounds = tuple((0, 1) for asset in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
        
    # def minimum_variance_portfolio(self, daily_returns):
    #     # Compute the covariance matrix
    #     cov_matrix = np.cov(daily_returns, rowvar=False)
        
    #     # Calculate the inverse of the covariance matrix
    #     inv_cov_matrix = np.linalg.inv(cov_matrix)
        
    #     # Create a vector of ones
    #     ones_vector = np.ones(len(cov_matrix))
        
    #     # Calculate the weights of the minimum variance portfolio
    #     weights = inv_cov_matrix.dot(ones_vector) / ones_vector.dot(inv_cov_matrix).dot(ones_vector)
        
    #     return weights
    
        # # Calculate the covariance matrix of the daily returns
        # covariance_matrix = daily_returns.cov().values
        
        # # The number of assets in the portfolio
        # num_assets = daily_returns.shape[1]
        
        # # Objective function to minimize (portfolio variance)
        # def portfolio_variance(weights):
        #     return np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # # Constraints: sum of weights = 1
        # constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # # Boundaries for weights: each weight can be between 0 and 1
        # bounds = tuple((0, 1) for asset in range(num_assets))
        
        # # Initial guess (equal weighting)
        # initial_guess = np.array([1. / num_assets] * num_assets)
        
        # # Optimization process
        # result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # # Check if the optimization was successful
        # if not result.success:
        #     raise BaseException("Optimization did not converge")
        
        # # Return the optimal weights
        # return result.x
    

    def calculate_weights_mdp(self, window=120):
        n_periods = len(self.data) - window + 1
        
        for start_index in range(0, n_periods):
            end_index = start_index + window
            # Extract the window of log returns
            windowed_data = self.log_ret.iloc[start_index:end_index]
            # Calculate covariance matrix for the window
            cov_matrix = windowed_data.cov().values
            # Calculate individual asset volatilities
            volatilities = np.sqrt(np.diag(cov_matrix))
            # Calculate MDP weights
            weights = self.max_diversification_portfolio(cov_matrix, volatilities)
            # Store the weights
            if end_index < len(self.data):
                self.weights.iloc[end_index] = weights
            else:
                self.weights.iloc[-1] = weights
            
            print(f"{start_index}-th iteration done")

        # Normalize weights to ensure they sum to 1 (and handle any NaN by forward filling then backfilling)
        self.weights = self.weights.div(self.weights.sum(axis=1), axis=0).ffill().bfill()
    
    def max_diversification_portfolio(self, cov_matrix, volatilities):
        num_assets = len(cov_matrix)
        initial_guess = np.ones(num_assets) / num_assets
        
        def objective(weights):
            # The numerator of the diversification ratio
            weighted_vol = np.dot(weights, volatilities)
            # The denominator of the diversification ratio
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # We maximize the negative of the diversification ratio (since we're using a minimization function)
            return -weighted_vol / portfolio_vol
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # The sum of weights must be equal to 1
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    # def maximum_diversification(daily_returns):
        # def MDP_pos_weights(self, datetime_mask, window = 120):
        #     #change the -1 in time window
        #     self.vols = np.sqrt( 250 / window * np.square(self.log_ret).rolling(window-1).sum() )
        #     self.weights = pd.DataFrame(data = [], index = self.data[datetime_mask].index, columns = self.data.columns)
            
        #     #start dates and end dates used in comatrice computation
        #     t_start = pd.DatetimeIndex(np.roll(self.data.index.values, window))[datetime_mask].dropna()
        #     t_end = self.data.index[datetime_mask].dropna()
        #     for i, date in enumerate(t_end):
        #         if t_end[i] < t_start[i]:
        #             pass
        #         else:
        #             ADMM = model.ADMM_Solver(self.data)
        #             ADMM.get_cov(by_hand = False, window = window, date = date)
        #             x,y = ADMM.solve(nb_iter = 10000, nb_iter_grad = 100, alpha = 0.0001, rho = 0.0001, random_permutation = True, lambda_star = 0.001, verbose = False)
        #             self.weights.loc[self.weights.index == date] = ((x+y)/2).reshape(1, 10)
        #         print(str(i) +"-th iteration done")
        #     self.weights = self.weights.dropna()
        #     self.weights = self.weights.div(self.weights.sum(axis=1), axis=0)
        #     self.weights = self.weights.reindex(self.data.index)
        #     self.weights = self.weights.ffill()
    # def maximum_diversification(self, daily_returns):
    #     # Calculate the covariance matrix of the daily returns
    #     covariance_matrix = daily_returns.cov()
        
    #     # Calculate the volatility (standard deviation) for each asset
    #     volatilities = daily_returns.std()
        
    #     # Define the objective function to minimize (negative diversification ratio)
    #     def objective(weights):
    #         # Portfolio volatility
    #         portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    #         # Weighted average volatility
    #         weighted_volatilities = np.dot(weights, volatilities)
    #         # Diversification ratio (we want to maximize it, so we minimize its negative)
    #         diversification_ratio = -weighted_volatilities / portfolio_volatility
    #         return diversification_ratio
        
    #     # Constraints: sum of weights = 1
    #     constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
    #     # Boundaries for weights: can't be less than 0 or more than 1
    #     bounds = tuple((0, 1) for asset in range(daily_returns.shape[1]))
        
    #     # Initial guess (equal weighting)
    #     initial_guess = np.array([1. / daily_returns.shape[1]] * daily_returns.shape[1])
        
    #     # Optimization
    #     result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
    #     # Return the optimal weights
    #     return result.x
        # # Compute the covariance matrix
        # cov_matrix = np.cov(daily_returns, rowvar=False)
        
        # # Compute the vector of variances
        # variances = np.diag(cov_matrix)
        
        # # Compute the inverse of the vector of variances
        # inv_variances = 1.0 / variances
        
        # # Compute the weights of the maximum diversification portfolio
        # weights = inv_variances / np.sum(inv_variances)
        
        # return weights
        

    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''

        # self.running_price_paths.append(asset_prices, ignore_index = True)

        # #Calculate sharpe ratios
        daily_returns = self.running_price_paths.pct_change().dropna()

        # # Calculate expected portfolio returns (R_p)
        # # Assuming weights is a numpy array of portfolio weights for the assets
        # expected_portfolio_return = np.dot(daily_returns.mean(), weights)

        # # Risk-free rate (R_f)
        # risk_free_rate = 0.01

        # weights = np.zeros(6)
        
        # # weights = np.array([0,1,-1,0.5,0.1,-0.2])
        # if self.max_diversification:
        #     # execute max diversification
        #     # Calculate covariance matrix
        #     min_variance = self.minimum_variance_portfolio(daily_returns)
            
            
        # else:
        #     # execute min variance
        #     # Calculate covariance matrix
        #     cov_matrix = daily_returns.cov()
        #     # Calculate inverse of covariance matrix
        #     inv_cov_matrix = np.linalg.inv(cov_matrix)
        #     # Calculate weights
        #     weights = np.dot(inv_cov_matrix, expected_returns) / np.sum(np.dot(inv_cov_matrix, expected_returns))
        
        #Calculate RPA
        # cov_matrix = daily_returns.cov()
        # rpa = np.dot(weights, np.dot(cov_matrix, weights))/np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))



        # #Code for 75% of array 1 and 25% of array 2 into new array
        # # Calculate the number of elements to take from array2 (25%)
        # num_elements_from_array2 = int(array1.size * 0.25)

        # # Generate random indices to replace with values from array2
        # indices_to_replace = np.random.choice(array1.size, num_elements_from_array2, replace=False)

        # # Create a new array from array1
        # new_array = np.copy(array1)

        # # Replace the selected indices in the new array with values from array2
        # new_array[indices_to_replace] = array2[indices_to_replace]
        
        # # Calculate the covariance matrix of the daily returns
        covariance_matrix = daily_returns.cov()
        # #Calculate volatilities
        # volatilities = daily_returns.std()
        # weights = self.max_diversification_portfolio(covariance_matrix, volatilities)
        # # weights = self.minimum_variance_portfolio(daily_returns)
        # # cov_matrix = daily_returns.cov()
        # # weights = np.array([0,1,-1,0.5,0.1,-0.2])
        # weights = np.dot(weights, np.dot(covariance_matrix, weights))/np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))

        volatilities = np.sqrt(np.diag(covariance_matrix))
        inverse_volatility = 1 / volatilities
        raw_weights = inverse_volatility / np.sum(inverse_volatility)

        # 'raw_weights' now represents a simple risk-parity inspired allocation
        weights = raw_weights

        mvp_weights = self.min_variance_portfolio(covariance_matrix)
        weights = mvp_weights
        
        weights = np.array([0.167, 0.2, 0.5, 0.02, 0.07, 0.6])
        #weights = self.minimum_variance_portfolio(daily_returns)

        return weights

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
