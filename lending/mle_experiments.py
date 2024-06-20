#Basic MLE Tutorial: https://analyticsindiamag.com/maximum-likelihood-estimation-python-guide/
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api
from scipy import stats
from scipy.optimize import minimize

# 1 Generate an independent variable
x = np.linspace(-10, 30, 100)
# generate a normally distributed residual
e = np.random.normal(10, 5, 100)
# generate ground truth
y = 10 + 4*x + e
df = pd.DataFrame({'x':x, 'y':y})
print(df.head())

# 2 Visualize Data
sns.regplot(x='x', y='y', data = df)
plt.show()

# 3 Make OLS Model
features = api.add_constant(df.x)
model = api.OLS(y, features).fit()
print(model.summary())

res = model.resid
standard_dev = np.std(res)
print(standard_dev)

# 4 ml modeling and neg LL calculation
def MLE_Norm(parameters):
    # extract parameters
    const, beta, std_dev = parameters
    # predict the output
    pred = const + beta*x
    # Calculate the log-likelihood for normal distribution
    LL = np.sum(stats.norm.logpdf(y, pred, std_dev))
    # Calculate the negative log-likelihood
    neg_LL = -1*LL
    return neg_LL

# 5 minimize arguments: function, intial_guess_of_parameters, method
mle_model = minimize(MLE_Norm, np.array([2,2,2]), method='L-BFGS-B')
print(mle_model)
