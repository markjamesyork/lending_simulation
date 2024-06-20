# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:24:43 2020

@author: USER
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def two_rec_two_bor(n):
    '''This simulation tests whether the marginal threshold of max(c_min, min.zero)
    creates incentive compatibility in the 2 recommender, 2 potential borrower,
    budget 1 scenario. This simulation assumes that recommender 1 rates truthfully,
    and that recommender 2 sees these recommendations before making his / her rating.
    '''
    c = .5 #Threshold aggregate rating which borrower must exceed to receive a loan
    rec_1_beliefs = np.random.rand(n,2)
    rec_2_beliefs = np.random.rand(n,2)
    rec_1_beliefs = np.asarray([[0.96644913, 0.70440981]])
    rec_2_beliefs = np.asarray([[0.80445814,	0.99540443]])
    honest_payout = []
    lying_payout = []
    
    for i in range(rec_1_beliefs.shape[0]):
        c_min = [min(2*c - j, .999) for j in rec_1_beliefs[i,:]]
        min_zero = np.asarray(([rec_1_beliefs[i,1] - rec_1_beliefs[i,0], \
                                rec_1_beliefs[i,0] - rec_1_beliefs[i,1]]))
        print('c_min',c_min)
        print('min_zero',min_zero)
        cij = [max(j,k) for j,k in zip(c_min, min_zero)]
        print('cij',cij)
        
        #Honest Payout
        if np.average([rec_1_beliefs[i,0], rec_2_beliefs[i,0]]) > \
                    np.average([rec_1_beliefs[i,1], rec_2_beliefs[i,1]]): winner = 0
        else: winner = 1
        p = rec_2_beliefs[i,winner]
        p_hat = rec_2_beliefs[i,winner]
        if np.average([rec_1_beliefs[i,winner], rec_2_beliefs[i,winner]]) < c: honest_payout += [0.]
        else:
            honest_payout += [p*(math.log(p_hat) - math.log(cij[winner])) \
                        / -math.log(cij[winner]) + (1-p)*(math.log(1-p_hat) - \
                        math.log(1 - cij[winner])) / -math.log(cij[winner])]
        #print('winner',winner)
        #Lying payout
        winner = (winner + 1) % 2
        p = rec_2_beliefs[i,winner]
        p_hat = np.max([p, cij[winner]]) #you prefer to rate truthfully, unless you must rate higher to get your favorite borrower allocated
        #print('p, p_hat',p,p_hat)
        lying_payout += [p*(math.log(p_hat) - math.log(cij[winner])) \
                    / -math.log(cij[winner]) + (1-p)*(math.log(1-p_hat) - \
                    math.log(1 - cij[winner])) / -math.log(cij[winner])]
        
    payout_difference = [i - j for i,j in zip(honest_payout, lying_payout)]
    is_below_zero = [i < 0. for i in payout_difference]
    #print('is_below_zero',is_below_zero)
    
    print('Pct of cases where lying is beneficial: ', 100 * sum(list(is_below_zero))/n, '%')
    print('min_payout_difference',min(payout_difference))
    print('payout_difference',payout_difference)
    print('honest_payout',honest_payout)
    print('lying_payout',lying_payout)
    print('rec_1_beliefs',rec_1_beliefs)
    print('rec_2_beliefs',rec_2_beliefs)
    return

#two_rec_two_bor(1)

def prop_4(n):
    '''This simulation tests whether a scoring rule which equally compensates the
    recommender at the critical threshold will be proper in all cases'''
    c = .5 #Threshold aggregate rating which borrower must exceed to receive a loan
    rec_1 = [.7,.4]
    rec_1_beliefs = np.tile(rec_1, (n,1)) #This and the line above make constant beliefs for recommender 1
    rec_2_beliefs = np.asarray([[.3,.6 + i*.01] for i in range(40)])
    #rec_2_beliefs = np.asarray([[0.3, .6],[0.4, .7],[0.5, .8],[0.6, .9],[0.7, 1.]])
    #rec_2_beliefs = np.asarray([[0.6, .9]])
    #rec_1_beliefs = np.random.rand(n,2)
    #rec_2_beliefs = np.random.rand(n,2)
    #rec_2_beliefs = np.asarray([[ 0.51261992,  0.11808637],[ 0.43005709,  0.36259722]\
    #                            , [ 0.33534227,  0.58978894]])
    honest_payout = []
    lying_payout = []
    
    for i in range(rec_1_beliefs.shape[0]):
        c_min = [min(2*c - j, .999) for j in rec_1_beliefs[i,:]]
        min_zero = np.asarray(([rec_1_beliefs[i,1] - rec_1_beliefs[i,0], \
                                rec_1_beliefs[i,0] - rec_1_beliefs[i,1]]))
        #print('c_min',c_min)
        #print('min_zero',min_zero)
        cij = [max(j,k) for j,k in zip(c_min, min_zero)]
        #print('cij',cij)
        c_dif = np.abs(cij[0] - cij[1])
        
        #Honest Payout
        if np.average([rec_1_beliefs[i,0], rec_2_beliefs[i,0]]) > \
                    np.average([rec_1_beliefs[i,1], rec_2_beliefs[i,1]]): winner = 0
        else: winner = 1
        p = rec_2_beliefs[i,winner]
        p_hat = rec_2_beliefs[i,winner]
        if np.average([rec_1_beliefs[i,winner], rec_2_beliefs[i,winner]]) < c: #checks whether beliefs are so low that neither borrower can get a loan
            honest_payout += [0.]
            lying_payout += [0.]
            continue
        else:
            if cij[winner] == max(cij):
                shift = c_dif
                cor = expected_score_winkler_shift(p_hat, p_hat, min(cij), shift)\
                            - expected_score_winkler_shift(p_hat-shift, p_hat-shift,\
                            min(cij))
            else:
                shift = 0. #shift needed to move the expected score in the higher-recommender case
                cor = 0. #correction added to the end to cut down the slope of the higher-threshold borrower to that of the lower-threshold borrower

            honest_payout += [expected_score_winkler_shift(p, p_hat, min(cij),\
                            shift) - cor]

        #Lying payout
        winner = (winner + 1) % 2
        p = rec_2_beliefs[i,winner]
        p_hat = np.max([p, cij[winner]]) #you prefer to rate truthfully, unless you must rate higher to get your favorite borrower allocated
        if cij[winner] == max(cij):
            shift = c_dif
            cor = expected_score_winkler_shift(p_hat, p_hat, min(cij), shift)\
                        - expected_score_winkler_shift(p_hat-shift, p_hat-shift,\
                        min(cij))
        else: 
            shift = 0. #shift needed to move the expected score in the higher-recommender case
            cor = 0. #correction added to the end to cut down the slope of the higher-threshold borrower to that of the lower-threshold borrower
        lying_payout += [expected_score_winkler_shift(p, p_hat, min(cij),\
                        shift) - cor]
        '''
        print('min(cij)',min(cij))
        print('score before cor',expected_score_winkler_shift(p, p_hat, min(cij),\
                        shift))
        print('winner',winner)
        print('p',p)
        print('p_hat',p_hat)
        print('shift',shift)
        print('cor',cor)
        '''
    payout_difference = [i - j for i,j in zip(honest_payout, lying_payout)]
    is_below_zero = [i < 0. for i in payout_difference]
    #print('is_below_zero',is_below_zero)
    
    print('Pct of cases where lying is beneficial: ', 100 * sum(list(is_below_zero))/n, '%')
    print('min_payout_difference',min(payout_difference))
    print('payout_difference',payout_difference)
    plt.plot(rec_2_beliefs[:,1],payout_difference)
    plt.xlabel('Rec 2 Belief on Borrower 2')
    plt.ylabel('Expected Payout to Recommender 2')
    plt.title('Expected Payout vs. Truthfully Reported Belief')
    print('dx',[payout_difference[i+1] - payout_difference[i] for i in range(39)])
    '''
    print('payout_difference',payout_difference)
    print('honest_payout',honest_payout)
    print('lying_payout',lying_payout)
    print('rec_1_beliefs',rec_1_beliefs)
    print('rec_2_beliefs',rec_2_beliefs)
    '''
    return

def expected_score_winkler_shift(p, p_hat, cij, shift = 0.):
    score = p*(math.log(p_hat - shift) - math.log(cij)) \
                / -math.log(cij) + (1-p)*(math.log(1-p_hat + shift) - \
                math.log(1 - cij)) / -math.log(cij)
    return score

def vcg_s(n,m,budget):
    '''This simulation tests our vcg allocation and payment rules in the budget-limited case
    We assume the ratings matrix is not sparse
    n = number of recommenders
    m = number of borrowers
    We assume that c=0, and we use the quadratic scoring rule for this simulation
    '''
    results = np.zeros((1,2))
    c = .0
    p = np.random.rand(n,m)
    '''
    p = np.asarray([[ 0.74830408,  0.2257707,   0.7287431 ],\
                    [ 0.43342216,  0.94162697,  0.98281962],\
                    [ 0.54284564,  0.43725587, 0.14753635]])
    '''
    #print('p',p)
    #p = np.asarray([[.5,.7],[.7,.6]])
    
    #Truthful Payout Calculation
    X = np.zeros(p.shape) #Matrix of allocations without each recommender, one by one
    for i in range(p.shape[0]):
        X[i,:] = wdp_s(c, np.delete(p, i, axis=0), budget) #Call WDP with each row deleted, one-by-one
    x = wdp_s(c, p, budget) #actual rating-maximizing allocations
    #print('X',X)
    #print('x',x)
    S = expected_score_quad(p, p, np.zeros(p.shape))
    #print('S1',S)
   # print('score sums',np.sum(S,axis=0))
    
    cij = cij_calc_s(p, S, X, x, budget)
    #print('cij',cij)
    cmin = cij.min(axis=1).reshape(p.shape[0],1) #row-wise minimums of cij
    p_mod = np.maximum(p - cij + cmin, np.asarray(p.shape[0]*[p.shape[1]*[.0]]))
    delta = expected_score_quad(p, p, cij) - expected_score_quad(p_mod, p_mod, cmin)
    S = expected_score_quad(p, p, cij) - delta
   # print('S2',S)
    expected_scores = np.matmul(S,x.T)
    results[0,0] = np.matmul(S,x.T)[-1]
        
    #print('Truthful Expected Payout',expected_scores)
    
    #Lying Payout Calculation
    #Setting lying p_hat for the last recommender
    p_hat = p.copy()
    marg_score = np.sort(p[-1,:])[-budget] #last recommender's belief for their budget-th highest borrower
    marg_index = np.argwhere(p_hat[-1,:] >= marg_score)
    p_hat[-1,:] = np.asarray([.0]*p.shape[1])
    for i in range(len(marg_index)):
        p_hat[-1,marg_index[i]] = p[-1,marg_index[i]]
    #print('p_hat',p_hat)
    X = np.zeros(p.shape) #Matrix of allocations without each recommender, one by one
    for i in range(p.shape[0]):
        X[i,:] = wdp_s(c, np.delete(p_hat, i, axis=0), budget) #Call WDP with each row deleted, one-by-one
    x = wdp_s(c, p_hat, budget) #actual rating-maximizing allocations
    #print('X',X)
    #print('x',x)
    S = expected_score_quad(p_hat, p_hat, np.zeros(p.shape))
    #print('S1',S)
    #print('score sums',np.sum(S,axis=0))
    
    cij = cij_calc_s(p_hat, S, X, x, budget)
    #print('cij',cij)
    cmin = cij.min(axis=1).reshape(p.shape[0],1) #row-wise minimums of cij
    print('cmin',cmin)
    p_mod = np.maximum(p_hat - cij + cmin, np.asarray(p.shape[0]*[p.shape[1]*[.0]]))
    delta = expected_score_quad(p_hat, p_hat, cij) - expected_score_quad(p_mod, p_mod, cmin)
    S = expected_score_quad(p, p_hat, cij) - delta
    #print('S2_lying',S)
    expected_scores = np.matmul(S,x.T)
    
    results[0,1] = np.matmul(S,x.T)[-1]
        
    #print('Lying Expected Payout',expected_scores)
    
    #Last Recommender Lying Payout Calculation
    '''This section assumes that the last recommender rates their top borrower truthfully and all other borrowers 0'''
    '''
    p_hat = p.copy()
    max_index = np.argwhere(p_hat[-1,:] == np.max(p_hat[-1,:]))[0][0]
    p_hat[-1,:] = np.asarray([.01]*p.shape[1])
    p_hat[-1,max_index] = p[-1,max_index]
    cij = cij_calc(p_hat, budget, c)
    S_true = expected_score_batch(p, p_hat, cij)
    S_reports = expected_score_batch(p_hat, p_hat, cij)
    x = wdp(c, p_hat, budget)
    expected_scores_true = np.matmul(S_true,x.T)
    expected_scores_reports = np.matmul(S_reports,x.T)
    lying_payout = np.zeros((1,n))
    
    for i in range(n): ###
        p_tmp = np.delete(p_hat, i, 0)
        cij_tmp = cij_calc(p_tmp, budget, c)
        S_tmp = expected_score_batch(p_tmp, p_tmp, cij_tmp)
        x_tmp = wdp(c, p_tmp, budget)
        lying_payout[0,i] = np.sum(expected_scores_reports) - np.sum(np.matmul(S_tmp, x_tmp.T))
        if i == n-1:
            lying_payout[0,i] = lying_payout[0,i] - np.sum(expected_scores_reports[i,0])\
                        + np.sum(expected_scores_true[i,0])
        
    print('Rec 2 Lying Expected Payout', lying_payout)
    '''
    return results

def cij_calc_s(p, S, X, x, budget):
    cij = np.zeros(p.shape)
    if np.sum(x) < budget: return cij
    for i in range(p.shape[0]):
        score_delta = np.sum(np.delete(np.matmul(S,x.T),i, axis=0)) - \
                    np.sum(np.delete(np.matmul(S,X[i,:].T),i, axis=0))
        #print('score_delta',score_delta)
        if score_delta < 0:
            differences = x - X[i,:] #1 where agent i causes a new borrower to be allocted, -1 where agent 1 causes a borrower not to be allocated
            #print('differences',differences)
            locs = np.argwhere(differences == 1)
            #print('locs',locs)
            delta_per_item = score_delta / locs.shape[0]
            #print('delta_per_item',delta_per_item)
            for j in range(locs.shape[0]):
                cij[i,locs[j][1]] = p[i,locs[j][1]] - (S[i,locs[j][1]] + delta_per_item)**.5
    return cij

def wdp_s(c, p, budget):
    S = expected_score_quad(p, p, np.zeros(p.shape))
    #ratings = p.sum(axis=0)
    ratings = S.sum(axis=0)
    threshold = max(np.sort(ratings)[-budget], c*p.shape[0])
    x = np.zeros((1,p.shape[1]))
    for i in range(p.shape[1]):
        if ratings[i] >= threshold: x[0,i] = 1
    return x

def cij_calc_p(p, budget, c):
    #p, S, X, x, budget
    cij = np.zeros(p.shape)
    if np.sum(x) < budget: return cij
    for i in range(p.shape[0]):
        score_delta = np.sum(np.delete(np.matmul(S,x.T),i, axis=0)) - \
                    np.sum(np.delete(np.matmul(S,X[i,:].T),i, axis=0))
        #print('score_delta',score_delta)
        if score_delta < 0:
            differences = x - X[i,:] #1 where agent i causes a new borrower to be allocted, -1 where agent 1 causes a borrower not to be allocated
            #print('differences',differences)
            locs = np.argwhere(differences == 1)
            #print('locs',locs)
            delta_per_item = score_delta / locs.shape[0]
            #print('delta_per_item',delta_per_item)
            for j in range(locs.shape[0]):
                cij[i,locs[j][1]] = p[i,locs[j][1]] - (S[i,locs[j][1]] + delta_per_item)**.5
    
    
    
    c_min = np.maximum(p + p.shape[0]*c - p.sum(axis=0), np.asarray(p.shape[0]*[p.shape[1]*[.001]])) #value which each recommender must rate each borrower for him/her to exceed c
    sums = p.sum(axis=0)
    min_zero = np.asarray(p.shape[0]*[p.shape[1]*[.001]])
    for i in range(p.shape[0]):
        sort = np.sort(sums - p[i,:]) #average scores sorted in reverse order
        ###Finding highest vulnerable borrower
        j = 0
        if budget > 1:
            vulnerable_value = sort[-budget-1] + 1
            while sort[-budget+j+1] < vulnerable_value and j + 1 < budget: j += 1
        #min_zero[i,:] = np.maximum(sort[-budget] - (sums - p[i,:]), np.asarray([.001]*p.shape[1])) #Gives value needed for target borrower to be top rated if target recommender rates all other borrowers at 0    
        min_zero[i,:] = np.maximum(sort[-budget+j] - (sums - p[i,:]), np.asarray([.001]*p.shape[1])) #Gives value needed for target borrower to be top rated if target recommender rates all other borrowers at 0    
    cij = np.minimum(np.maximum(c_min, min_zero), np.asarray(p.shape[0]*[p.shape[1]*[.999]]))
    return cij

def expected_score_quad(p, p_hat, cij):
    S = (1/(2-4*cij + 2*cij**2))*(p*(2*p_hat - 2*cij) + (1-p)*(2*(1-p_hat) - 2*(1-cij))\
                 - p_hat**2 - (1-p_hat)**2 + cij**2 + (1-cij)**2)
    S = np.where(cij >= .9999, 0, S) #corrects for cases with extremely high cij, when loans should not be allocated
    S = np.where(p_hat < cij, 0, S) #truncates the expected score at zero when the report is below the threshold
    return S

def expected_score_batch(p, p_hat, cij):
    S = p*(np.log(p_hat) - np.log(cij)) \
                / -np.log(cij) + (1-p)*(np.log(1-p_hat) - \
                np.log(1 - cij)) / -np.log(cij)
    S = np.where(cij >= .999, 0, S) #corrects for cases with extremely high cij, when loans should not be allocated
    S = np.where(p_hat < cij, 0, S) #truncates the expected score at zero when the report is below the threshold
    return S

def wdp_p(cij, p, budget, c):
    ratings = p.sum(axis=0)
    threshold = max(np.sort(ratings)[-budget], c*p.shape[0])
    meets_thresh = (ratings > threshold)
    x = np.zeros((1,p.shape[1]))
    for i in range(p.shape[1]):
        if meets_thresh[i]: x[0,i] = 1
    return x

def vcg_p(n,m,budget):
    '''This simulation tests our vcg allocation and payment rules in the budget-limited case
    We assume the ratings matrix is not sparse, and we use vcg with respect to
    ratings (p). The lost p to other recommenders is charged to a randomly-selected set of borrowers
    n = number of recommenders
    m = number of borrowers
    '''
    results = np.zeros((1,2))
    c = 0.
    p = np.random.rand(n,m)
    #p = np.asarray([[.5,.7,.9],[.5,.7,.9],[.7,.6,.7],[.9,.51,.1]]) #Breaking Example
    print('p',p)
    
    #Truthful Payout Calculation  
    cij = cij_calc_p(p, budget, c)
    x = wdp_p(c, p, budget)
    if np.sum(x) > 0: #Run this if at least one borrower receives a loan
        print('cij',cij)
        cmin = cij.min(axis=1).reshape(p.shape[0],1) #row-wise minimums of cij
        p_mod = np.maximum(p - cij + cmin, np.asarray(p.shape[0]*[p.shape[1]*[.001]]))
        #print('d1', expected_score_batch(p, p, cij))
        #print('d2', expected_score_batch(p_mod, p_mod, cmin))
        delta = expected_score_batch(p, p, cij) - expected_score_batch(p_mod, p_mod, cmin)
        #print('delta',delta)
        S = expected_score_batch(p, p, cij) - delta
        #print('S',S)
        print('x_true',x)
        print('Truthful Expected Payout',np.matmul(S,x.T))
        truthful = np.matmul(S,x.T)
        results[0,0] = np.matmul(S,x.T)[-1]

        #Last Recommender Lying Payout Calculation
        '''This section assumes that the last recommender rates their top borrower truthfully and all other borrowers 0'''
        p_hat = p.copy()
        marg_score = np.sort(p[-1,:])[-budget] #last recommender's belief for their budget-th highest borrower
        marg_index = np.argwhere(p_hat[-1,:] >= marg_score)
        p_hat[-1,:] = np.asarray([.001]*p.shape[1])
        #print('marg_index',marg_index)
        for i in range(len(marg_index)):
            p_hat[-1,marg_index[i]] = p[-1,marg_index[i]]
        #print('p_hat',p_hat)
        cij = cij_calc_p(p_hat, budget, c)
        x = wdp_p(cij, p_hat, budget)
        cmin = cij.min(axis=1).reshape(p.shape[0],1) #row-wise minimums of cij
        p_mod = np.maximum(p_hat - cij + cmin, np.asarray(p.shape[0]*[p.shape[1]*[.001]]))
        delta = expected_score_batch(p, p, cij) - expected_score_batch(p_mod, p_mod, cmin)
        S = expected_score_batch(p, p_hat, cij) - delta
        print('x_lying',x)
        print('Last Rec Lying Expected Payout', np.matmul(S,x.T))
        print('Truthful - lying: ', (truthful - np.matmul(S,x.T))[-1])
        results[0,1] = np.matmul(S,x.T)[-1]
        return results

    else:
        #print('x',x)
        #print('Truthful Expected Payout',np.zeros((p.shape[0],1)))
        #print('Lying Expected Payout',np.zeros((p.shape[0],1)))
        return np.zeros((1,2))

vcg_p(3,3,2)
#vcg_s(5,5,3)

def call_vcg_s(num):
    results = np.zeros((num,3))
    for i in range(num):
        tmp = vcg_p(10,10,8)
        results[i,:2] = tmp
        results[i,2] = results[i,0] - results[i,1]
    print('results',results)
    print('Max Lying Advantage: ',-np.min(results[:,2]))
    print('Average Truth Telling Advantage: ', np.mean(results[:,2]))
    print('Average Truthful Payout: ', np.mean(results[:,0]))
    is_above_thresh = [i < -0.02 for i in results[:,2]]
    print('Pct of cases where lying is beneficial: ', 100 * sum(list(is_above_thresh))/num, '%')

    return

#call_vcg_s(100)