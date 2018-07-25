# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:03:22 2018

@author: Vivek Mishra
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np
result = pd.read_csv('results.csv')

#Cut off by date
result['date'] = pd.to_datetime(result['date'])
result = result.set_index(['date'])
start = datetime.strptime("2006-01-01","%Y-%m-%d").date()
end = datetime.strptime("2018-12-01","%Y-%m-%d").date()
result = result[start:end]
#result = result.drop(result[result.tournament == 'Friendly'].index)

#Get data for only world cup qualified team
mega = pd.DataFrame()

worldCup_nation = ['Argentina','Australia','Belgium','Brazil','Colombia','Costa Rica',
                   'Croatia','Denmark','Egypt','England','France','Germany',
                   'Iceland','Iran','Japan','Mexico','Morocco','Nigeria','Panama',
                   'Peru','Poland','Portugal','Russia','Saudi Arabia','Senegal',
                   'Serbia','Korea Republic','Spain','Switzerland','Sweden','Tunisia','Uruguay']



#mega = result.drop(result[(not result.home_team.isin(worldCup_nation)) & (not result.away_team.isin(worldCup_nation))].index)
mega = result[(result['home_team'].isin(worldCup_nation)) | (result['away_team'].isin(worldCup_nation))]

#Reading ELO Ranking
elo = pd.DataFrame()
for i in range(1999,2019):
    data = pd.read_csv(str(i)+'.TSV', sep='\t',header=None)
    data['year'] = i
    elo = elo.append(data)
elo = elo.dropna(axis=0)    
#Name convention
country_name = pd.read_csv("en_sanity.TSV", sep='\t',header=None)  

def sanity_name(code):
    print(code)
    country = country_name[country_name[0] == code][1]
    return country.values[0]

elo[2] = elo[2].apply(lambda x: sanity_name(x))

#Column names for ELo
elo = elo[[1,2,3,22,26,27,28,29,30,'year']]
elo.columns  = ['rank','country','points','total','wins','loss','draw','for','against','year']
elo['winp'] = elo['wins']/elo['total']
elo['gd'] = elo['for'] - elo['against']

elo = elo.replace('Serbia and Montenegro', 'Serbia')


#Merge ELO and result
def target_var(home,away):
    if home > away:
        return 1
    elif home < away:
        return 0
    else:
        return 2

mega['target'] = mega.apply(lambda row: target_var(row['home_score'], row['away_score']), axis=1)
mega = mega.drop(['home_score','away_score'], axis=1)
mega = mega.reset_index()

def rankELO(date,country,sub=1):
    year = date.year - sub
    #print(str(country)+str(year))
    rank = elo[(elo['year'] == year) & (elo['country'] == country)]['rank']
    #print(rank)
    if not rank.empty:
        return rank.values[0]
    else:
        return 250

def pointsELO(date,country,sub=1):
    year = date.year - sub
    #print(str(country)+str(year))
    points = elo[(elo['year'] == year) & (elo['country'] == country)]['points']
    if not points.empty:
        return points.values[0]
    else:
        return 0

def winpELO(date,country,sub=1):
    year = date.year-sub
    winp = elo[(elo['year'] == year) & (elo['country'] == country)]['winp']
    if not winp.empty:
        return winp.values[0]
    else:
        return 0

def gdELO(date,country,sub=1):
    year = date.year-sub
    gd = elo[(elo['year'] == year) & (elo['country'] == country)]['gd']
    if not gd.empty:
        return gd.values[0]
    return 0


#Drop certain countries due to missing ranking
mega = mega.drop(mega[mega.home_team == 'Burma'].index)
mega = mega.drop(mega[mega.home_team == 'Namibia'].index)
mega = mega.drop(mega[mega.away_team == 'Burma'].index)
mega = mega.drop(mega[mega.away_team == 'Namibia'].index)


mega['home_rank'] = 0
mega['home_rank'] = mega.apply(lambda row: rankELO(row['date'],row['home_team']), axis=1)

mega['away_rank'] = 0
mega['away_rank'] = mega.apply(lambda row: rankELO(row['date'],row['away_team']), axis=1)

mega['home_points'] = 0
mega['home_points'] = mega.apply(lambda row: pointsELO(row['date'],row['home_team']), axis=1)

mega['away_points'] = 0
mega['away_points'] = mega.apply(lambda row: pointsELO(row['date'],row['away_team']), axis=1)

mega['home_winp'] = 0
mega['home_winp'] = mega.apply(lambda row: winpELO(row['date'],row['home_team']), axis=1)

mega['away_winp'] = 0
mega['away_winp'] = mega.apply(lambda row: winpELO(row['date'],row['away_team']), axis=1)

mega['home_gd'] = 0
mega['home_gd'] = mega.apply(lambda row: gdELO(row['date'],row['home_team']), axis=1)

mega['away_gd'] = 0
mega['away_gd'] = mega.apply(lambda row: gdELO(row['date'],row['away_team']), axis=1)


mega.neutral = mega.neutral.astype(int)
#Let's start modelling


#Training set
mega['date'] = pd.to_datetime(mega['date'])
mega = mega.set_index(['date'])
mega = mega.sort_index()

import copy
train = copy.deepcopy(mega)
train = train.drop(['tournament','city','country'], axis=1)
train = train.drop(['home_team','away_team'], axis=1)
train['rank_difference'] = train['home_rank'] - train['away_rank']
train['points_difference'] = train['home_points'] - train['away_points']
train['winp_difference'] = train['home_winp'] - train['away_winp']
train['gd_difference'] = train['home_gd'] - train['away_gd']
train = train.drop(['home_rank','away_rank','home_points','away_points','home_gd','away_gd','home_winp','away_winp'], axis=1)
train_start = datetime.strptime("2006-01-01","%Y-%m-%d").date()
train_end = datetime.strptime("2018-06-30","%Y-%m-%d").date()
train = train[train_start:train_end]




X_train = train.drop('target', axis=1).values
y_train = train['target'].values
y_train = y_train.reshape(-1,1)
y_train = np.ravel(y_train)

###############
#XGBOOST
import xgboost as xgb 


dtrain = xgb.DMatrix(X_train, y_train)

#default parameters
params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'multi:softprob',
}
params['eval_metric'] = "merror"
params['num_class'] = 3
num_boost_round = 999

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(1,8)
    for min_child_weight in range(1,6)
]
min_merror = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))

    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )

    # Update best MError
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (max_depth,min_child_weight)


params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]
        
#tune subsample,colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(1,11)]
    for colsample in [i/10. for i in range(1,11)]
]  

min_merror = float("Inf")
best_params = None
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))

    # Update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )

    # Update best Merror
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (subsample,colsample)
        
        
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]


min_merror = float("Inf")
best_params = None
for eta in [0.5,0.3, 0.03, .003,0.0003]:
    print("CV with eta={}".format(eta))

    # Update our parameters
    params['eta'] = eta

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )

    # Update best Merror
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = eta
        
params['eta'] = best_params

#Test Set
test = mega
test = test.drop(['tournament','city','country'], axis=1)
test = test.drop(['home_team','away_team'], axis=1)
test['rank_difference'] = test['home_rank'] - test['away_rank']
test['points_difference'] = test['home_points'] - test['away_points']
test['winp_difference'] = test['home_winp'] - test['away_winp']
test['gd_difference'] = test['home_gd'] - test['away_gd']
test = test.drop(['home_rank','away_rank','home_points','away_points','home_gd','away_gd','home_winp','away_winp'], axis=1)
test_start = datetime.strptime("2017-01-01","%Y-%m-%d").date()
test_end = datetime.strptime("2018-06-01","%Y-%m-%d").date()
test = test[test_start:test_end]
X_test = test.drop('target', axis=1).values
y_test = test['target'].values


xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 1
param['min_child_weight'] = 1
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 3
param['colsample_bytree'] = 1.0
param['subsample'] = 1.0
param['gamma'] = 0.1


watchlist = [(xg_train, 'train')]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != y_test) / y_test.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
pred_prob = bst.predict(xg_test).reshape(y_test.shape[0], 3)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != y_test) / y_test.shape[0]
print('Test error using softprob = {}'.format(error_rate))

from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(bst)
pyplot.show()


#Accuracy
def error(actual,pred):
    if actual == pred:
        return 1
    else:
        return 0

actual = mega
actual = actual[test_start:test_end]
actual['pred'] = pred
actual['accuracy'] = actual.apply(lambda row: error(row['target'],row['pred']), axis=1)
actual['row_no'] = list(range(0, 712))

accuracy = actual["accuracy"].sum()/len(actual["accuracy"])
print("model accuracy"+str(accuracy))





#Prediction:    
match_date = datetime.strptime("2018-06-14","%Y-%m-%d").date()
world_cup = pd.read_csv('World Cup 2018 Dataset.csv')
world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
world_cup = world_cup.dropna(how='all')
world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "Korea Republic"})
world_cup = world_cup.set_index('Team')    



from itertools import combinations

opponents = ['First match \nagainst', 'Second match\n against', 'Third match\n against']
world_cup['points'] = 0
world_cup['total_prob'] = 0
for group in set(world_cup['Group']):
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
        print("Home: {} vs. Away: {}: ".format(home, away), end='')
        #Prepare our predict set
        predict = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,
                                          np.nan]]), columns=['neutral','home_rank','away_rank','home_points','away_points','home_winp','away_winp','home_gd','away_gd'])
        if home == 'Russia' or away == 'Russia':
            predict['neutral'] = 0
        else:
            predict['neutral'] = 1
        
        predict['home_rank'] = rankELO(match_date,home,0)
        
        predict['away_rank'] = rankELO(match_date,away,0)
        
        predict['home_points'] = pointsELO(match_date,home,0)
        
        predict['away_points'] = pointsELO(match_date,away,0)
        
        predict['home_winp'] = winpELO(match_date,home, 0)
        
        predict['away_winp'] = winpELO(match_date,away, 0)
        
        predict['home_gd'] = gdELO(match_date,home, 0)
        
        predict['away_gd'] = gdELO(match_date,away, 0)
        
        predict['rank_difference'] = predict['home_rank'] - predict['away_rank']
        predict['points_difference'] = predict['home_points'] - predict['away_points']
        predict['winp_difference'] = predict['home_winp'] - predict['away_winp']
        predict['gd_difference'] = predict['home_gd'] - predict['away_gd']
        predict = predict.drop(['home_rank','away_rank','home_points','away_points','home_gd','away_gd','home_winp','away_winp'], axis=1)
        
        
        xg_test = xgb.DMatrix(predict.values)
        shape = predict['neutral'].values   
        pred_prob = bst.predict(xg_test).reshape(1, 3)
        pred_label = np.argmax(pred_prob, axis=1)
        
        points = 0
        
        if pred_label[0] == 0:
            world_cup.loc[away, 'points'] += 3
        elif pred_label[0] == 2:
            world_cup.loc[home, 'points'] += 1
            world_cup.loc[away, 'points'] += 1
        elif pred_label[0] == 1:
            world_cup.loc[home, 'points'] += 3
            
        print("Home win prob: "+str(pred_prob[0][1])+" Away win prob: "+str(pred_prob[0][0])+" Draw: "+str(pred_prob[0][2]))
    


#Knockout

pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]
world_cup = world_cup.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
next_round_wc = world_cup.groupby('Group').nth([0, 1]) # select the top 2
next_round_wc = next_round_wc.reset_index()
next_round_wc = next_round_wc.loc[pairing]
next_round_wc = next_round_wc.set_index('Team')    

finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']


for f in finals:
    print("___Starting of the {}___".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []
    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home,away), end='')
        
        predict = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,
                                          np.nan]]), columns=['neutral','home_rank','away_rank','home_points','away_points','home_winp','away_winp','home_gd','away_gd'])
        if home == 'Russia' or away == 'Russia':
            predict['neutral'] = 0
        else:
            predict['neutral'] = 1
        
        predict['home_rank'] = rankELO(match_date,home,0)
        
        predict['away_rank'] = rankELO(match_date,away,0)
        
        predict['home_points'] = pointsELO(match_date,home,0)
        
        predict['away_points'] = pointsELO(match_date,away,0)
        
        predict['home_winp'] = winpELO(match_date,home, 0)
        
        predict['away_winp'] = winpELO(match_date,away, 0)
        
        predict['home_gd'] = gdELO(match_date,home, 0)
        
        predict['away_gd'] = gdELO(match_date,away, 0)
        
        predict['rank_difference'] = predict['home_rank'] - predict['away_rank']
        predict['points_difference'] = predict['home_points'] - predict['away_points']
        predict['winp_difference'] = predict['home_winp'] - predict['away_winp']
        predict['gd_difference'] = predict['home_gd'] - predict['away_gd']
        predict = predict.drop(['home_rank','away_rank','home_points','away_points','home_gd','away_gd','home_winp','away_winp'], axis=1)
        
        xg_test = xgb.DMatrix(predict.values)
        shape = predict['neutral'].values   
        pred_prob = bst.predict(xg_test).reshape(1, 3)
        pred_label = np.argmax(pred_prob, axis=1)
        
        if pred_label[0] == 0:
            winners.append(away)
        elif pred_label[0] == 2:
            if pred_prob[0][1] > pred_prob[0][0]:
                winners.append(home)
            else:
                winners.append(away)
        elif pred_label[0] == 1:
            winners.append(home)
        
        
        print("Home win prob: "+str(pred_prob[0][1])+" Away win prob: "+str(pred_prob[0][0])+" Draw: "+str(pred_prob[0][2]))
        
    next_round_wc = next_round_wc.loc[winners]
    print("\n")






