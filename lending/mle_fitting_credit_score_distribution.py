#MLE To Fit Distribution for Credit Scores
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

#1 Data
percentiles = [.1, .25, .5 ,.75, .9]
scores = [548, 634, 729, 798, 822]
scores_norm = [(x - 350)/500 for x in scores] #normalizes credit scores so that 350 = 0 and 850 = 1
#Source: https://www.federalreserve.gov/econres/notes/feds-notes/developments-in-the-credit-score-distribution-over-2020-accessible-20210430.htm#fig2

#2 Beta Fitting w/o MLE
def sse_beta(parameters):
    a = parameters[0]
    b = parameters[1]
    pred = [stats.beta.ppf(i, a, b) for i in percentiles] #ppf is inverse of cdf
    squared_errors = [(x-y)**2 for x, y in zip(scores_norm, pred)]
    print('pred, scores', pred, scores)
    return np.sum(squared_errors)

beta_fit = minimize(sse_beta, np.array([2,2]), method='L-BFGS-B')

#3 Print Results
print('model', beta_fit)
model_preds = [stats.beta.ppf(i, beta_fit.x[0], beta_fit.x[1]) for i in percentiles]
model_scores = [350 + 500*i for i in model_preds]
print('model_scores', model_scores)
print('scores', scores)
print('model_preds',model_preds)
print('scores_norm', scores_norm)
#Beta parameters found: [2.5036631 , 1.01396259]


'''
#2 Beta fitting
def MLE_Beta(parameters):
    # extract parameters
    print('parameters',parameters)
    a = parameters[0]
    b = parameters[1]
    std_dev = parameters[2]

    # predict the output
    pred = [stats.beta.cdf(x, a, b) for x in percentiles] #prediction of what the normalized credit score should be at each point in the CDF given the alpha and beta chosen

    # Calculate the log-likelihood for normal distribution
    LL = np.sum(stats.norm.logpdf(scores_norm, pred, std_dev))

    # Calculate the negative log-likelihood
    neg_LL = -1*LL
    return neg_LL

mle_beta_fit = minimize(MLE_Beta, np.array([2,2,.1]), method='L-BFGS-B')
print(mle_beta_fit)
'''
