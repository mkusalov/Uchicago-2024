import numpy as np
import pandas as pd
import scipy

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

prices_df = pd.read_csv('Training Data_Case 3.csv', index_col=0)


returns_df = prices_df.pct_change().dropna()
COV_matrix = returns_df.cov()
weighty = np.array([0,1,-1,0.5,0.1,-0.2])
   

def window_COV(window_size):
    global COV_matrix
    global returns_df

    recent_df = returns_df.tail(window_size)
    COV_matrix = recent_df.cov()

def rebalance(weights):
    for i in range(len(weights)):
        if weights[i] < 0:
            weights[i] = 0
    weights = weights/np.sum(weights)
    return weights

def ewma_cov_pairwise_pd(x, y, alpha=0.06):
    x = x.mask(y.isnull(), np.nan)
    y = y.mask(x.isnull(), np.nan)
    covariation = ((x - x.mean()) * (y - y.mean()).dropna())
    return covariation.ewm(alpha=0.06).mean().iloc[-1]

def ewma_cov_pd(rets, alpha=0.06):
    assets = rets.columns
    n = len(assets)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = cov[j, i] = ewma_cov_pairwise_pd(
                rets.iloc[:, i], rets.iloc[:, j], alpha=alpha)
    return pd.DataFrame(cov, columns=assets, index=assets)

def allocate_portfolio(asset_prices):
    global weighty
    global COV_matrix
    global returns_df

    one = np.array([1,1,1,1,1,1,1,1,1,1])
    prices_df.loc[len(prices_df.index)+1] = asset_prices
    returns_df = prices_df.pct_change().dropna()
    
    window_COV(130)
    global_minimum_weights1 = np.matmul(np.linalg.pinv(COV_matrix),one)/np.matmul(one.T,np.matmul(np.linalg.pinv(COV_matrix),one)).item(0)
    weighty1 = global_minimum_weights1

    COV_matrix = ewma_cov_pd(returns_df,alpha=0.06)
    global_minimum_weights2 = np.matmul(np.linalg.pinv(COV_matrix),one)/np.matmul(one.T,np.matmul(np.linalg.pinv(COV_matrix),one)).item(0)
    weighty2 = global_minimum_weights2

    weight_comb = rebalance(((.75)*weighty1 + (.25)*weighty2))
    weighty = weight_comb
    return weight_comb


def grading(testing): #testing is a pandas dataframe with price data, index and column names don't matter
    weights = np.full(shape=(len(testing.index),10), fill_value=0.0)
    for i in range(0,len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i,:])))
        positive = np.absolute(unnormed)
        normed = positive/np.sum(positive)
        weights[i]=list(normed)
    capital = [1]
    for i in range(len(testing) - 1):
        shares = capital[-1] * np.array(weights[i]) / np.array(testing.iloc[i,:])
        capital.append(float(np.matmul(np.reshape(shares, (1,10)),np.array(testing.iloc[i+1,:]))))
    returns = (np.array(capital[1:]) - np.array(capital[:-1]))/np.array(capital[:-1])
    return np.mean(returns)/ np.std(returns) * (252 ** 0.5), capital, weights