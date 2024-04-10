import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

pricex  s = pd.read_csv('Training Data_Case 3.csv', index_col=0)
returns_data = prices.pct_change().dropna()

train_split = 0
training_threshold = int(len(prices) * train_split)

training_prices = prices
running_prices = training_prices
training_returns = returns_data
running_returns = training_returns

expanding_100_day_returns = training_returns.iloc[-100:]

max_sharpe_sharpes = []
hrp_sharpes = []

new_prices = pd.DataFrame(data=training_prices.iloc[-5:], columns=training_prices.columns)

updateDays = 0.0
trainingWeights = [2.09141447e-18, 2.27179160e-01, 2.34292027e-01, 2.16047525e-01, 7.20937235e-03, 3.40358085e-18,
                   0.00000000e+00, 1.46155663e-01, 1.69116253e-01, 5.02468754e-19]
trainingWeights = np.array(trainingWeights)


# =====================================================================
# Markowitz Optimization
# =====================================================================
def expectedReturn(asset_prices):
    returns = np.diff(asset_prices, axis=0) / asset_prices[0:len(asset_prices) - 1]
    return returns


def covarianceMatrix(expectedReturns):
    covariance_matrix = np.cov(expectedReturns, rowvar=False)
    return covariance_matrix


def markowitzWeights(mu_vec, cov_matrix, rf_rate=0):
    rf_vec = np.full(mu_vec.shape, rf_rate)
    mu_diff_rf = mu_vec - rf_vec
    np.expand_dims(mu_diff_rf, axis=1)
    inverseCov = np.linalg.inv(cov_matrix)
    unnormalized = np.matmul(mu_diff_rf, inverseCov)
    for i in range(0, len(unnormalized)):
        if unnormalized[i] < 0:
            unnormalized[i] = 0
    normalize = unnormalized / np.linalg.norm(unnormalized, 1)
    return normalize


def maxSharpe(mu_vec, cov_matrix, rf_rate=0):
    global prices

    def sharpe_helper(weights):
        returns = np.mean(mu_vec, axis=0)
        port_ret = returns @ weights
        port_std = np.sqrt(weights.transpose() @ cov_matrix @ weights)
        return -1 * port_ret / port_std

    def weight_constraint(weights):
        return np.sum(weights) - 1

    bounds_lim = [(0, 1) for x in range(prices.shape[1])]
    init = np.ones(prices.shape[1]) / prices.shape[1]
    constraint = {'type': 'eq', 'fun': weight_constraint}
    optimal = scipy.optimize.minimize(fun=sharpe_helper,
                                      x0=init,
                                      bounds=bounds_lim,
                                      constraints=constraint,
                                      method='SLSQP'
                                      )

    return optimal['x']


# =====================================================================
# Hierarchical Risk Parity 
# =====================================================================
def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0], axis=0)
        sort_ix = sort_ix.sort_index()

    return sort_ix.tolist()


def get_inverse_variance_weights(cov_matrix):
    ivp = 1. / np.diag(cov_matrix)
    ivp /= ivp.sum()
    return ivp


def hierarchical_risk_parity(returns):
    corr_matrix = returns.corr()
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)

    link = linkage(squareform(dist_matrix), method='single')

    sorted_idx = get_quasi_diag(link)
    sorted_corr_matrix = corr_matrix.iloc[sorted_idx, sorted_idx]

    weights = pd.Series(index=sorted_corr_matrix.index, dtype=float)
    for i in range(len(sorted_corr_matrix)):
        cov_matrix = sorted_corr_matrix.iloc[:i + 1, :i + 1]
        weight_slice = get_inverse_variance_weights(cov_matrix)
        weights.iloc[i] = weight_slice[-1]

    return weights / weights.sum()


def calc_sharpe(returns):
    return np.mean(returns) / np.std(returns) * (252 ** 0.5)


def calc_rsi(ts):
    price_diffs = np.diff(ts)
    gains = np.where(price_diffs > 0, price_diffs, 0)
    losses = np.where(price_diffs < 0, -price_diffs, 0)

    avg_gain = np.sum(gains[:14]) / 14
    avg_loss = np.sum(losses[:14]) / 14

    rsi = 100 - (100 / (1 + (14 - (avg_gain / avg_loss))))
    return rsi


def calc_returns(weighted_returns):
    cumulative_returns = np.concatenate(([1], (1 + weighted_returns).cumprod()))
    capital = cumulative_returns
    returns = (np.array(capital[1:]) - np.array(capital[:-1])) / np.array(capital[:-1])
    return returns


def allocate_portfolio(asset_prices):
    global updateDays
    global trainingWeights, running_prices, training_prices, training_returns, running_returns, new_prices, \
        expanding_100_day_returns, max_sharpe_sharpes, hrp_sharpes

    asset_prices = pd.DataFrame(data=[asset_prices], columns=running_prices.columns, index=[len(running_prices)])
    running_prices = pd.concat([running_prices, asset_prices])

    current_returns = running_prices.pct_change().iloc[-1].to_frame().T
    running_returns = pd.concat([running_returns, current_returns])
    expanding_100_day_returns = pd.concat([expanding_100_day_returns, current_returns])

    new_prices = pd.concat([new_prices, asset_prices])

    # Markowitz/Max Sharpe Weights
    updateDays += 1.0
    returns = expectedReturn(new_prices)
    covariance = covarianceMatrix(returns)
    mu_vec = np.mean(returns, axis=0) * 252
    testingWeights = maxSharpe(returns, covariance)
    testEmphasis = updateDays / (updateDays + len(training_prices))
    maxSharpeWeights = (testEmphasis * testingWeights) + ((1 - testEmphasis) * trainingWeights)

    weighted_daily_returns = (maxSharpeWeights * expanding_100_day_returns).sum(axis=1)
    max_sharpe_returns = calc_returns(weighted_daily_returns)
    max_sharpe_sharpe = calc_sharpe(max_sharpe_returns)
    max_sharpe_sharpes.append(max_sharpe_sharpe)

    hrp_weights = hierarchical_risk_parity(running_returns).sort_index()
    weighted_daily_returns = (hrp_weights * expanding_100_day_returns).sum(axis=1)
    hrp_returns = calc_returns(weighted_daily_returns)
    hrp_sharpe = calc_sharpe(hrp_returns)
    hrp_sharpes.append(hrp_sharpe)

    m = 3
    max_sharpe_emphasis = (np.exp(max_sharpe_sharpe * m)) / ((np.exp(max_sharpe_sharpe * m)) + (np.exp(hrp_sharpe * m)))
    hrp_emphasis = (np.exp(hrp_sharpe * m)) / ((np.exp(max_sharpe_sharpe * m)) + (np.exp(hrp_sharpe * m)))

    combined_weights = (hrp_emphasis * hrp_weights) + (max_sharpe_emphasis * maxSharpeWeights)
    return combined_weights


def grading(testing):  # testing is a pandas dataframe with price data, index and column names don't matter
    weights = np.full(shape=(len(testing.index), 10), fill_value=0.0)
    for i in range(0, len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i, :])))
        positive = np.absolute(unnormed)
        normed = positive / np.sum(positive)
        weights[i] = list(normed)
    capital = [1]
    for i in range(len(testing) - 1):
        shares = capital[-1] * np.array(weights[i]) / np.array(testing.iloc[i, :])
        capital.append(float(np.matmul(np.reshape(shares, (1, 10)), np.array(testing.iloc[i + 1, :]))))
    returns = (np.array(capital[1:]) - np.array(capital[:-1])) / np.array(capital[:-1])
    return np.mean(returns) / np.std(returns) * (252 ** 0.5), capital, weights