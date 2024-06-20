#Gamma Distribution Experiments
#Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

#Calculate the first four moments:
a = 20
b = .5
mean, var, skew, kurt = gamma.stats(a, scale = b, moments='mvsk')

#Display the probability density function (pdf):
x = np.linspace(gamma.ppf(0.01, a, scale = b),
                gamma.ppf(0.99, a, scale = b), 100)
ax.plot(x, gamma.pdf(x, a, scale = b),
       'r-', lw=5, alpha=0.6, label='gamma pdf')

#Freeze the distribution and display the frozen pdf
rv = gamma(a, scale = b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

#Check accuracy of cdf and ppf
vals = gamma.ppf([0.001, 0.5, 0.999], a, scale = b)
np.allclose([0.001, 0.5, 0.999], gamma.cdf(vals, a, scale = b))

#Generate random numbers:
r = gamma.rvs(a, scale = b, size=1000)

#And compare the histogram:
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
