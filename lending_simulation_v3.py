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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import rankdata
import time


def main(n_recommenders, n_borrowers, budget, c, pct_dishonest, \
            rec_alpha_plus_beta, mean_borrower_repayment):
    #inputs = recommenders, borrowers, budget, lending threshold c, percent of
    #recommenders who are honest, alpha + beta of recommender knowledge (higher = tighter accuracy)
    #mean borrower repayment probability

    #This function simulates the truncated-winkler or 0-1 VCG for n_rounds rounds

    #0 Parameters
    '''
    #2021 Feb recommender knowledge parameters
    alpha = .75 #This represents the quality of recommenders' knowledge of borrower repayment probability. Alpha is the weight of the true probabilities, and 1-alpha is the weight of random noise in recommender beliefs
    p = alpha*np.tile(repayment_probs, (n_recommenders, 1)) + \
            (1-alpha)*np.random.random((n_recommenders, n_borrowers)) #recommender beliefs - |N| x |M|
    '''
    mechanism = 'vcg_scoring' #'truncated_winkler', 'vcg_scoring'
    shift_a = 10**-3
    shift_b = 10**9
    shift = beta.rvs(shift_a, shift_b, size=n_recommenders)*.5 #The upward bias of each recommender
    rec_accuracy_weight_vector = gamma.rvs(a = rec_alpha_plus_beta*2, scale = .5,\
                size=n_recommenders) #Total weight alpha + beta to use in the beta distribution of recommender beliefs
    n_rounds = 1
    mean_repayment_prob = mean_borrower_repayment #alpha / (alpha + beta) for beta distribution
    borrower_alpha_plus_beta = 9.9 #9.9 = Fit from Uganda recommendations data. alpha + beta for beta distribution of repayment probs. Higher values will concentrate probs around the mean


    #0.1 Honesty Parameters
    honesty_type = 'collusion' #{'honest', 'random_misreports', 'misreports_select_recommenders', 'collusion'}
    frac_misreports = 1 #fraction of reports which are misreports under 'random_misreports' honesty type. In honesty type 'misreports_select_recommenders', the fraction of reports by dishonest recommenders which are misreports. In honesty type 'collusion', this is the fraction of all potential borrowers on whom dishonest recommenders misreport. Note this number is rounded down to the nearest integer number of borrowers.
    frac_misreports_1 = .75 #fraction of misreports which are 1 in all honesty types. All misreports other than 1 are 0.
    frac_dishonest_recommenders = pct_dishonest #fraction of recommenders who are dishonest in 'misreports_select_recommenders' and 'collusion' honesty types

    if honesty_type in ['misreports_select_recommenders', 'collusion']: #Set which recommenders will be colluding in all rounds
        '''#Deterministic Colluder Setting: exactly pct_dishonest fraction of borrowers will collude every time
        rand_rec_ranks = rankdata(np.random.random(n_recommenders))
        misreporting_recommenders = np.where(rand_rec_ranks <= n_recommenders*\
                    frac_dishonest_recommenders, 1, np.zeros((1,n_recommenders)))[0] #vector with 1s for misreporting recommenders and 0 for honest recommenders
        '''
        #Random Collusion: each borrower will independently at random become a colluder with prob pct_dishonest
        misreporting_recommenders = np.where(np.random.random(n_recommenders) < \
                    pct_dishonest, 1, 0)
        #print('misreporting_recommenders: ', misreporting_recommenders)

    #0.3 Creating lists where round-by-round data will be stored
    n_loans_made = []
    n_repayments = []
    mean_comp = []
    std_comp = []
    rec_comp_negative_pct = []
    rounds_to_low_wts_for_dishonest_recs = n_rounds #This variable will be changed later IF there is a round where all dishonest recommenders are given low weights AND collusion type \in {'misreports_select_recommenders', 'collusion'}

    for i in range(n_rounds):
        if (i == 0 or np.sum(multi_round_lending_decisions) == 0): \
                    weights = np.ones(n_recommenders)/n_recommenders #Use equal weights for all recommenders
        else: weights = generate_weights(reports, multi_round_lending_decisions,\
                        multi_round_outcomes, n_recommenders, n_borrowers)
        #print('weights: ', np.round(weights,2))
        if (rounds_to_low_wts_for_dishonest_recs == n_rounds and \
                    np.max(np.multiply(weights, misreporting_recommenders)) < .01):
            rounds_to_low_wts_for_dishonest_recs = i #means that no dishonest recommenders have a weight above the threshold during the i+1th round of lending observations


        #1 Borrow Prob and Recommender Belief Creation
        repayment_probs = beta.rvs(mean_repayment_prob*borrower_alpha_plus_beta,\
                    borrower_alpha_plus_beta*(1-mean_repayment_prob), size=n_borrowers) #True repayment probability for each borrower
        central_probs = np.clip(np.tile(repayment_probs, (n_recommenders,1)) + \
                        np.tile(shift, (n_borrowers,1)).T, .01, .99)
        alphas = np.multiply(central_probs, np.tile(rec_accuracy_weight_vector,\
                    (n_borrowers,1)).T)
        betas = np.tile(rec_accuracy_weight_vector,\
                    (n_borrowers,1)).T - alphas
        p = beta.rvs(alphas, betas, size=(n_recommenders, n_borrowers))


        #2 Reporting / Misreporting Mechanism
        p_hat = p.copy() #truthful reporting

        if honesty_type == 'random_misreports':
            #A random subset of all reports are set to 0 or 1
            n_reports = n_recommenders * n_borrowers
            rand_ranks = rankdata(np.random.random(p.shape)).reshape(p.shape)
            p_hat = np.where(rand_ranks <= n_reports * frac_misreports * \
                     frac_misreports_1, 1, p_hat)
            p_hat = np.where((rand_ranks > n_reports * frac_misreports * \
                     frac_misreports_1) & (rand_ranks <= \
                     n_reports * frac_misreports), 0, p_hat)

        elif honesty_type == 'misreports_select_recommenders':
            #A subset of recommenders each chooses a random set of borrowers on whom to missreport. The other recommenders are honest.
            for j in range(n_recommenders):
                if misreporting_recommenders[j] == 0.: continue #skip misreport setting for honest recommenders
                rand_bor_ranks = rankdata(np.random.random(n_borrowers))
                p_hat[j] = np.where(rand_bor_ranks <= n_borrowers*\
                            frac_misreports \
                            * frac_misreports_1, 1, p_hat[j])
                p_hat[j] = np.where((rand_bor_ranks > n_borrowers*\
                            frac_misreports \
                            * frac_misreports_1) & (rand_bor_ranks <= \
                            n_borrowers*frac_misreports), 0, p_hat[j])

        elif honesty_type == 'collusion':
            #A subset of recommenders together chooses a random set of borrowers on whom they will coordinate misreports. The other recommenders are honest.
            #Note that "collusion" has no effect on whether borrowers repay.
            rand_bor_ranks = rankdata(np.random.random(n_borrowers))
            for j in range(n_borrowers):
                if rand_bor_ranks[j] > n_borrowers*frac_misreports:\
                            continue
                if rand_bor_ranks[j] <= n_borrowers*frac_misreports*\
                            frac_misreports_1:
                    p_hat[:,j] = np.where(misreporting_recommenders == 1, 1, p_hat[:,j])
                else:
                    p_hat[:,j] = np.where(misreporting_recommenders == 1, 0, p_hat[:,j])


        #3 Repayment and Recommender Compensation Calculation
        if mechanism == 'truncated_winkler':
                    expected_payout, actual_payout, lending_decisions, \
                    repayment_outcomes = truncated_winkler(p, p_hat, \
                    repayment_probs, c, weights)
        elif mechanism == 'vcg_scoring':
                    expected_payout, actual_payout, lending_decisions, \
                    repayment_outcomes = vcg(p, p_hat, \
                    repayment_probs, budget, c, weights)


        #4 Data Calculation & Storage
        if mechanism == 'truncated_winkler': comp_by_recommender = np.sum(actual_payout, axis=1)
        else: comp_by_recommender = actual_payout
        n_loans_made += [np.sum(lending_decisions)]
        n_repayments += [np.sum(repayment_outcomes)]
        mean_comp += [np.mean(comp_by_recommender)]
        std_comp += [np.std(comp_by_recommender)]
        rec_comp_negative_pct += [np.sum(np.where(comp_by_recommender < 0, 1,\
                            0)) / p.shape[0]]
        if i == 0:
            reports = p_hat
            multi_round_lending_decisions = lending_decisions
            multi_round_outcomes = repayment_outcomes
        else:
            reports = np.hstack((reports, p_hat))
            multi_round_lending_decisions = np.hstack((multi_round_lending_decisions, \
                        lending_decisions))
            multi_round_outcomes = np.hstack((multi_round_outcomes,repayment_outcomes))

    #string of parameter settings to pass for storage with results
    params = mechanism +','+ '%f_%f' % (shift_a, shift_b) +','+ honesty_type	+','+ str(frac_misreports)\
                +','+ str(frac_misreports_1) +','+ str(n_rounds)

    return np.asarray(n_loans_made), np.asarray(n_repayments), \
                np.asarray(mean_comp), np.asarray(std_comp), \
                np.asarray(rec_comp_negative_pct), [rounds_to_low_wts_for_dishonest_recs],\
                params #End of main()


def truncated_winkler(p, p_hat, repayment_probs, c, weights):
    #This function runs the truncated winkler scoring system with the given parameters
    base_rule = 'quadratic' #logarithmic option not yet programmed up.
    #0 Lending Decisions
    #print('repayment_probs',repayment_probs)

    #lending_decisions = np.where(p_hat.sum(axis=0) > c*p.shape[0], 1, 0) #w/o weights
    lending_decisions = np.where(np.matmul(weights.T, p_hat) > c, 1, 0)
    #print('lending_decisions',lending_decisions)

    #1.0 Calculate Thresholds ciq
    ciq = np.zeros(p.shape) #matrix of lending thresholds by recommender / borrower combo
    for i in range(p.shape[0]): #index over recommenders
        #ciq[i,:] = p.shape[0] * c - (p_hat.sum(axis=0) - p_hat[i]) #w/o weights
        if weights[i] < 10**-6: ciq[i,:] = np.ones(p.shape[1]) #this person has no weight and will not be able to decide who gets a loan.
        else:
            ciq[i,:] = (c - np.matmul(weights.T, p_hat) + weights[i]*p_hat[i,:])/weights[i]
    ciq = np.clip(ciq, 0.01, .99)
    #print('ciq',ciq)

    #1.5 Calculate expected scores for each recommender and borrower
    if base_rule == 'quadratic':
        scores_repay, scores_default = winkler_quadratic_scores(p_hat, ciq,\
                        lending_decisions)
    expected_payout = np.multiply(scores_repay, p) + np.multiply(\
                        scores_default, (1-p))

    #2 Repayment outcomes
    repayment_outcomes = np.multiply(np.where(np.random.random(\
                lending_decisions.shape) > (1-repayment_probs), 1, 0),\
                lending_decisions)
    actual_payout = np.multiply(scores_repay, repayment_outcomes) + \
                np.multiply(scores_default, (1-repayment_outcomes)) #payouts when all recommenders have weight 1
    actual_payout = np.multiply(actual_payout, \
                np.asarray(weights).reshape((len(weights),1))) #payouts scaled by recommender weight

    return expected_payout, actual_payout, lending_decisions, repayment_outcomes


def winkler_quadratic_scores(p_hat, ciq, lending_decisions):
    #Returns matrices of payouts in the case of repayment or default. 0 where
    #loans not made. Assumes all recommenders have weight 1 (we weight when calculating actual payout)
    denominator = 2 - 4*ciq + 2 * ciq*ciq
    scores_repay = (2*p_hat - p_hat*p_hat - (1-p_hat)*(1-p_hat) - 2*ciq + \
                    ciq*ciq + (1-ciq)*(1-ciq)) / denominator
    scores_default = (2*(1-p_hat) - p_hat*p_hat - (1-p_hat)*(1-p_hat) - \
                      2*(1-ciq) + ciq*ciq + (1-ciq)*(1-ciq)) / denominator

    return lending_decisions*scores_repay, lending_decisions*scores_default


def vcg(p, p_hat, repayment_probs, budget, c, weights):
    #This function runs the VCG scoring mechanism with a reserve
    #0 Setup
    n_recommenders = p.shape[0]
    n_borrowers = p.shape[1]
    #augment p_hat with reserve borrowers and reserve recommender
    p_hat = np.vstack((p_hat, np.zeros((1,n_borrowers))))
    tmp_array = np.vstack((np.zeros((n_recommenders,budget)), np.full((1,budget),\
                    c)))
    p_hat = np.hstack((p_hat, tmp_array))
    weights_aug = np.hstack((weights,np.asarray([1])))

    #1 Lending decisions
    #sums = np.sum(p_hat, axis = 0) #w/o weights
    sums = np.matmul(weights_aug, p_hat)
    sums.sort()
    threshold = sums[-budget]
    sums = np.matmul(weights_aug, p_hat)
    lending_decisions = np.where(sums[:n_borrowers] >= threshold, 1, 0) #Will not allocate any reserve borrowers yet
    lending_decisions = np.hstack((lending_decisions, np.zeros((budget,))))
    if np.sum(lending_decisions) > budget: #lending_decisions may allocate too many borrowers in the case that multiple borrowers are tied at the threshold
        lending_decisions = np.where(sums[:n_borrowers] > threshold, 1, 0)
        lending_decisions = np.hstack((lending_decisions, np.zeros((budget,))))
        ties_indices = np.argwhere(sums[:n_borrowers] == threshold)
        count = 0
        while np.sum(lending_decisions) < budget: #Add tied candidates until budget is met
            lending_decisions[ties_indices[count]] = 1
            count += 1
    elif np.sum(lending_decisions) < budget: #allocates just enough reserve borrowers to fill the budget quota
        lending_decisions[n_borrowers: int(n_borrowers + budget - \
                    np.sum(lending_decisions))] = 1

    #2 Expected Score
    payments = np.zeros(n_recommenders)
    expected_payout = np.zeros(n_recommenders)

    for j in range(n_recommenders): #loop over borrowers to calculate payment t and expected score
        p_hat_less_j = np.delete(p_hat, j, 0)
        weights_aug_less_j = np.delete(weights_aug, j)
        sums_j = np.matmul(weights_aug_less_j, p_hat_less_j)
        sums_j.sort()
        threshold = sums_j[-budget]
        sums_j = np.matmul(weights_aug_less_j, p_hat_less_j)
        lending_decisions_j = np.where(sums_j[:n_borrowers] >= threshold, 1, 0) #Will not allocate any reserve borrowers yet
        lending_decisions_j = np.hstack((lending_decisions_j, np.zeros((budget,))))
        if np.sum(lending_decisions_j) > budget: #lending_decisions may allocate too many borrowers in the case that multiple borrowers are tied at the threshold
            lending_decisions_j = np.where(sums[:n_borrowers] > threshold, 1, 0)
            lending_decisions_j = np.hstack((lending_decisions_j, np.zeros((budget,))))
            ties_indices = np.argwhere(sums_j[:n_borrowers] == threshold)
            count = 0
            '''
            print('sums_j',sums_j)
            print('lending_decisions_j',lending_decisions_j)
            print('ties_indices',ties_indices)
            print('threshold',threshold)
            '''
            while np.sum(lending_decisions_j) < budget: #Add tied candidates until budget is met
                lending_decisions_j[ties_indices[count]] = 1
                count += 1
        elif np.sum(lending_decisions_j) < budget: #allocates just enough reserve borrowers to fill the budget quota
            lending_decisions_j[n_borrowers: int(n_borrowers + budget - \
                             np.sum(lending_decisions_j))] = 1
        payments[j] = np.sum(np.multiply(sums_j, lending_decisions_j)) - \
                    np.sum(np.multiply(sums_j, lending_decisions)) #value to everyone else without i present - value to everyone else with i present
        expected_payout[j] = np.sum(np.multiply(p[j,:],\
                    lending_decisions[:n_borrowers]))*weights[j] - payments[j] #This is the expected payout from the recommender's perspective, assuming his/her beliefs are true

    #print('payments t:', np.round(payments,2))
    #print('expected_payout', np.round(expected_payout,2))
    #3 Repayment Outcomes
    repayment_outcomes = np.multiply(np.where(np.random.random(\
                n_borrowers) > (1-repayment_probs), 1, 0),\
                lending_decisions[:n_borrowers])
    actual_payout = np.sum(repayment_outcomes)*weights - payments
    #actual_payout = np.sum(repayment_outcomes) - payments
    #print('repayment_outcomes', repayment_outcomes)
    #print('actual_payout',np.round(actual_payout,2))

    return expected_payout, actual_payout, lending_decisions[:n_borrowers],\
                repayment_outcomes

def generate_weights(reports, multi_round_lending_decisions,
                     multi_round_outcomes, n_recommenders, n_borrowers):
    '''This function generates linear weights using the method by Budescu & Chen.
    reports is an n_recommenders  by rounds_observed * n_borrowers matrix
    multi_round_outcomes is a 1 by n_borrowers * rounds_observed matrix
    multi_round_lending_decisions is a 1 by n_borrowers * rounds_observed matrix
    '''
    '''
    print('reports: ', reports)
    print('multi_round_lending_decisions',multi_round_lending_decisions)
    print('multi_round_outcomes',multi_round_outcomes)
    print('n_recommenders',n_recommenders)
    print('n_borrowers',n_borrowers)
    '''
    mean_recommender_contributions = np.zeros(n_recommenders)
    aggregated_reports = np.sum(reports, axis=0) / n_recommenders #simple average of reports; we could use weighted reports
    group_scores = brier(aggregated_reports, multi_round_outcomes) #includes an outcome of 0 for borrowers who did not receive a loan
    mean_group_score = np.matmul(group_scores, multi_round_lending_decisions.T) \
                / np.sum(multi_round_lending_decisions)
    for i in range(n_recommenders):
        agg_reports_without_recommender = (np.sum(reports, axis=0) - reports[i,:])\
                    / (n_recommenders - 1)
        scores_without_recommender = brier(agg_reports_without_recommender, \
                    multi_round_outcomes)
        mean_recommender_contributions[i] = mean_group_score - \
                    np.matmul(scores_without_recommender, multi_round_lending_decisions.T) \
                / np.sum(multi_round_lending_decisions)
    positive_contributions = np.clip(mean_recommender_contributions, 0, 10**6)
    if np.sum(positive_contributions) > 0: #case where at least one person has made a postiive contribution
        weights = positive_contributions / np.sum(positive_contributions)
    else: #case where no one has made a positive contribution
        weights = np.ones(n_recommenders)/n_recommenders #Use equal weights for all recommenders

    #print('weights',weights)
    return weights

def brier(preds, outcomes):
    #This function takes a set of preds \in [0,1] for binary events and calculates the brier score against actual outcomes \in {0,1}
    preds_of_outcomes = 1 - preds - outcomes + 2*np.multiply(preds, outcomes) #predictions made of the outcomes that actually occurred
    return 2*preds_of_outcomes - np.power(preds,2) - np.power(1-preds,2)



##### CODE TO RUN THE SIMULATION #####
n_recommenders_values = list(np.arange(1, 21))
pct_dishonest_values = [0]*20
rec_alpha_plus_beta_values = [12.74]*20
mean_borrower_repayment_values = [.5]*20


for iter in range(len(pct_dishonest_values)):
    print('Iteration: ', iter)
    reps = 10**4 #Run the simulation 'reps' times to average across noise

    for k in range(reps):
        #print('Rep: ',k)
        #1 Parameter Settings
        n_recommenders = n_recommenders_values[iter]
        n_borrowers = 20
        budget = 3 #Maximum number of borrowers who can receive a loan
        c = .85 #Lending threshold rating
        pct_dishonest = pct_dishonest_values[iter] #percent of recommenders who are dishonest
        rec_alpha_plus_beta = rec_alpha_plus_beta_values[iter] #tightness of recommender knowledge in a beta distribution. Higher = better knowledge. When this parameter is 10, assuming 5 alpha and 5 beta, the std dev of the recommender knowledge is .15
        mean_borrower_repayment = mean_borrower_repayment_values[iter]

        #Standard temperature and pressure = 6 recommenders, 10 borrowers, budget 6, c = .8, pct_dishonest = .34, rec_alpha_plus_beta = 10, mean borrower repayment of .6

        #2 Simulation Loops
        if k == 0: n_loans_array, n_repayments_array, mean_recommender_comp_array,\
                    rec_comp_vol_array, rec_comp_negative_pct_array, \
                    rounds_to_low_wts_for_dishonest_recs_list, params = \
                    main(n_recommenders, n_borrowers, budget, c, pct_dishonest,\
                    rec_alpha_plus_beta, mean_borrower_repayment)
        else:
            n_loans_made, n_repayments, mean_comp, std_comp, rec_comp_negative_pct,\
                        rounds_to_low_wts_for_dishonest_recs, params\
                        = main(n_recommenders, n_borrowers, budget, c, pct_dishonest,\
                        rec_alpha_plus_beta, mean_borrower_repayment)
            n_loans_array = np.vstack((n_loans_array,n_loans_made))
            n_repayments_array = np.vstack((n_repayments_array,n_repayments))
            mean_recommender_comp_array = np.vstack((mean_recommender_comp_array,mean_comp))
            rec_comp_vol_array = np.vstack((rec_comp_vol_array,std_comp))
            rec_comp_negative_pct_array = np.vstack((rec_comp_negative_pct_array,rec_comp_negative_pct))
            rounds_to_low_wts_for_dishonest_recs_list += rounds_to_low_wts_for_dishonest_recs


    print('n_loans',np.round(np.mean(n_loans_array),2))
    print('repayment_rate',np.round(np.sum(n_repayments_array) / np.sum(n_loans_array),2))
    print('mean_recommender_comp',np.round(np.mean(mean_recommender_comp_array),2))
    print('rec_comp_vol',np.round(np.nanmean(rec_comp_vol_array),2))
    print('rec_comp_negative_pct',np.round(np.mean(rec_comp_negative_pct_array),2))
    print('rounds_to_low_wts_for_dishonest_recs',np.round(np.mean(rounds_to_low_wts_for_dishonest_recs_list),2))

    # Calculate lender profit with 20% interest
    profit_array = 1.2 * n_repayments_array - n_loans_array
    print('profit_array', np.mean(profit_array))

    row = '\n' + str(time.strftime('%Y%m%d')) +','+ str(n_recommenders) +','+ \
                str(n_borrowers) +','+ str(budget) +','+ str(c) +','+ str(pct_dishonest)\
                 +','+ str(rec_alpha_plus_beta) +','+ str(mean_borrower_repayment) \
                 +','+ params +','+ str(reps) +','+ str(np.round(np.mean(n_loans_array),3))\
                 +','+ str(np.round(np.sum(n_repayments_array) / np.sum(n_loans_array),3))\
                 +','+ str(np.round(np.mean(mean_recommender_comp_array),3)) +','+\
                 str(np.round(np.nanmean(rec_comp_vol_array),3)) +','+ \
                 str(np.round(np.mean(rec_comp_negative_pct_array),3)) +','+\
                 str(np.round(np.mean(rounds_to_low_wts_for_dishonest_recs_list),3)) +','+\
                 str(np.round(np.mean(profit_array / budget),3)) # Profit divided by budget

    with open('lending_simulation_results.csv','a') as fd:
        fd.write(row)
