# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:18:11 2021

@author: Mark York
"""

'''
This script defines a set of borrowers and a set of agents who have beliefs about those
borrowers. The system then simulates the outcomes of different rating strategies
for both recommenders and the principal in terms of repayments and recommender compensation
'''

import numpy as np

def main(n_recommenders, n_borrowers, budget, c):
    #inputs = recommenders, borrowers, budget, lending threshold c
    #The basic model is a classless matrix-based simulation of the truncated-winkler and 0-1 VCG solutions
    
    #0 Parameters
    alpha = .75 #This represents the quality of recommenders' knowledge of borrower repayment probability. Alpha is the weight of the true probabilities, and 1-alpha is the weight of random noise in recommender beliefs
    payout = '0_1_vcg'#'truncated_winkler'#
    
    #1 Variable Creation
    repayment_probs = np.random.random((n_borrowers)) #True repayment probability for each borrower
    p = alpha*np.tile(repayment_probs, (n_recommenders, 1)) + \
                (1-alpha)*np.random.random((n_recommenders, n_borrowers)) #recommender beliefs - |N| x |M|
    
    #2 Report & Expected Calculation
    p_hat = p.copy() #honest reporting
    
    #3 Repayment and Recommender Compensation Calculation   
    if payout == 'truncated_winkler':
                expected_payout, actual_payout, lending_decisions, \
                repayment_outcomes = truncated_winkler(p, p_hat, \
                repayment_probs, c)
    elif payout == '0_1_vcg':
                expected_payout, actual_payout, lending_decisions, \
                repayment_outcomes = vcg(p, p_hat, \
                repayment_probs, budget, c)
                
    #4 Lender economics calculation
    if payout == 'truncated_winkler': comp_by_recommender = np.sum(actual_payout, axis=1)
    else: comp_by_recommender = actual_payout
    rec_comp_negative_pct = np.sum(np.where(comp_by_recommender < 0, 1, 0)) / p.shape[0]
    
    #5 Printing
    '''
    print('lending_decisions',lending_decisions)         
    #print('expected_payout',expected_payout)
    
    #Results
    print('Results: ')
    repayment_rate = np.sum(repayment_outcomes) / np.sum(lending_decisions)
    #print('repayment_outcomes',repayment_outcomes)
    #print('actual_payout',actual_payout)
    print('repayment_rate',repayment_rate)
    print('mean_rec_comp_std',np.mean(comp_by_recommender))
    print('rec_comp_negative_pct',rec_comp_negative_pct)   
    '''         
    return np.sum(lending_decisions), np.sum(repayment_outcomes), np.mean(comp_by_recommender),\
                np.std(comp_by_recommender), rec_comp_negative_pct

def truncated_winkler(p, p_hat, repayment_probs, c):
    #This function runs the truncated winkler scoring system with the given parameters
    base_rule = 'quadratic' #options include 'quadratic' and 'logarithmic'
    #0 Lending Decisions
    #print('repayment_probs',repayment_probs)  
    
    lending_decisions = np.where(p_hat.sum(axis=0) > c*p.shape[0], 1, 0)
    #print('lending_decisions',lending_decisions)
    
    #1 Expected payout
    cij = np.zeros(p.shape) #matrix of lending thresholds by recommender / borrower combo
    for i in range(p.shape[0]):
        cij[i,:] = p.shape[0] * c - (p_hat.sum(axis=0) - p_hat[i])
    cij = np.clip(cij, 0.000001, .999999)
    if base_rule == 'quadratic':
        scores_repay, scores_default = winkler_quadratic_scores(p_hat, cij,\
                        lending_decisions)
    expected_payout = np.multiply(scores_repay, p) + np.multiply(\
                        scores_default, (1-p))
    
    #2 Repayment outcomes
    repayment_outcomes = np.multiply(np.where(np.random.random(\
                lending_decisions.shape) > (1-repayment_probs), 1, 0),\
                lending_decisions)
    actual_payout = np.multiply(scores_repay, repayment_outcomes) + \
                np.multiply(scores_default, (1-repayment_outcomes))

    return expected_payout, actual_payout, lending_decisions, repayment_outcomes

def winkler_quadratic_scores(p_hat, cij, lending_decisions):
    #Returns matrices of payouts in the case of repayment or default. 0 where
    #loans not made
    denominator = 2 - 4*cij + 2 * cij*cij
    scores_repay = (2*p_hat - p_hat*p_hat - (1-p_hat)*(1-p_hat) - 2*cij + \
                    cij*cij + (1-cij)*(1-cij)) / denominator
    scores_default = (2*(1-p_hat) - p_hat*p_hat - (1-p_hat)*(1-p_hat) - \
                      2*(1-cij) + cij*cij + (1-cij)*(1-cij)) / denominator
    return lending_decisions*scores_repay, lending_decisions*scores_default


def vcg(p, p_hat, repayment_probs, budget, c):
    #This function runs the 0-1 VCG mechanism with a reserve
    #0 Setup
    n_recommenders = p.shape[0]
    n_borrowers = p.shape[1]
    #augment p_hat with reserve borrowers and reserve recommender
    p_hat = np.vstack((p_hat, np.zeros((1,n_borrowers))))
    tmp_array = np.vstack((np.zeros((n_recommenders,budget)), np.full((1,budget),\
                    c*n_recommenders)))
    p_hat = np.hstack((p_hat, tmp_array))
    
    #1 Lending decisions
    sums = np.sum(p_hat, axis = 0)
    #print('sums',sums)
    sums.sort()
    threshold = sums[-budget]
    #print('sums.sort()',sums)
    sums = np.sum(p_hat, axis = 0)
    lending_decisions = np.where(sums[:n_borrowers] >= threshold, 1, 0) #Will not allocate any reserve borrowers yet
    lending_decisions = np.hstack((lending_decisions, np.zeros((budget,))))
    #print('threshold',threshold)
    #print('lending_decisions',lending_decisions)
    #print('p_hat',p_hat)
    

    if np.sum(lending_decisions) < budget: #allocates just enough reserve borrowers to fill the budget quota
        lending_decisions[n_borrowers: int(n_borrowers + budget - \
                    np.sum(lending_decisions))] = 1
    
    #2 Expected Score
    payments = np.zeros(n_recommenders)
    expected_payout = np.zeros(n_recommenders)
    for i in range(n_recommenders): #loop over borrowers to calculate payment t and expected score
        p_hat_less_i = np.delete(p_hat, i, 0)
        sums_i = np.sum(p_hat_less_i, axis = 0)
        sums_i.sort()
        threshold = sums_i[-budget]
        sums_i = np.sum(p_hat_less_i, axis = 0)
        lending_decisions_i = np.where(sums_i[:n_borrowers] > threshold, 1, 0) #Will not allocate any reserve borrowers yet
        lending_decisions_i = np.hstack((lending_decisions_i, np.zeros((budget,))))
        if np.sum(lending_decisions_i) < budget: #allocates just enough reserve borrowers to fill the budget quota
            lending_decisions_i[n_borrowers: int(n_borrowers + budget - \
                             np.sum(lending_decisions_i))] = 1
        payments[i] = np.sum(np.multiply(sums_i, lending_decisions_i)) - \
                    np.sum(np.multiply(sums_i, lending_decisions)) #value to everyone else without i present - value to everyone else with i present
        expected_payout[i] = np.sum(np.multiply(p[i,:],\
                    lending_decisions[:n_borrowers])) - payments[i] #This is the expected payout from the recommender's perspective, assuming his/her beliefs are true
        
    #3 Repayment Outcomes
    repayment_outcomes = np.multiply(np.where(np.random.random(\
                n_borrowers) > (1-repayment_probs), 1, 0),\
                lending_decisions[:n_borrowers])
    actual_payout = np.sum(repayment_outcomes) - payments

    return expected_payout, actual_payout, lending_decisions[:n_borrowers],\
                repayment_outcomes

#Run main n times
n = 1000
n_loans_vec = np.zeros(n)
repayment_rate_vec = np.zeros(n)
mean_recommender_comp_vec = np.zeros(n)
rec_comp_vol_vec = np.zeros(n)
rec_comp_negative_pct_vec = np.zeros(n)

for i in range(n):
    n_loans, repayment_rate, mean_recommender_comp, rec_comp_vol, \
            rec_comp_negative_pct = main(6, 10, 7, .5) #(n_recommenders, n_borrowers, budget, c)
    n_loans_vec[i] = n_loans
    repayment_rate_vec[i] = repayment_rate
    mean_recommender_comp_vec[i] = mean_recommender_comp
    rec_comp_vol_vec[i] = rec_comp_vol
    rec_comp_negative_pct_vec[i] = rec_comp_negative_pct
    
print('n_loans',np.mean(n_loans_vec))
print('repayment_rate',np.sum(repayment_rate_vec) / np.sum(n_loans_vec))
#print('repayment_rate_vec',repayment_rate_vec)
print('mean_recommender_comp',np.mean(mean_recommender_comp_vec))
print('rec_comp_vol',np.nanmean(rec_comp_vol_vec))
print('rec_comp_negative_pct',np.mean(rec_comp_negative_pct_vec))