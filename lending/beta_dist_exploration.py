#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 10:21:38 2021

@author: markyork
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
fig, ax = plt.subplots(1, 1)

#Moments
a, b = .5, 5
mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
print('moments - mvsk:', mean, var, skew, kurt )
#Note, it seems that the mean is always a / (a + b)

#Display pdf function
x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),'r-', lw=5, alpha=0.6, label='beta pdf')

#Freeze the Distribution
rv = beta(a, b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

#Check the accuracy of the pdf and cdf
vals = beta.ppf([0.001, 0.5, 0.999], a, b)
np.allclose([0.001, 0.5, 0.999], beta.cdf(vals, a, b))

#Generate random numbers
r = beta.rvs(a, b, size=1000)

#Compare the historgram
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()