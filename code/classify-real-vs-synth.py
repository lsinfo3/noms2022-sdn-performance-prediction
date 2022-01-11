# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:58:37 2021

@author: katha
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.feature_selection import SelectPercentile
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score


arch = "HF"
if (arch == "KN"):
    df = pd.read_csv('../data/data_KN.csv', sep=";").fillna(value=-1)
if (arch == "HF"):
    df = pd.read_csv('../data/data_HF.csv', sep=";").fillna(value=-1)
    df = df[df.columns.drop(list(df.filter(regex='C2RL')))]

df['Real']=1

df['Network'].replace(to_replace='[0-9]*', value='',inplace=True,regex=True) 
df['Network'].replace(to_replace='KentmanFeb|KentmanJan', value='Kentman',inplace=True,regex=True) 


if (arch == "KN"):
    df_rnd = pd.read_csv('../data/data_rnd_KN.csv', sep=";").fillna(value=-1)
if (arch == "HF"):
    df_rnd = pd.read_csv('../data/data_rnd_HF.csv', sep=";").fillna(value=-1)
    df_rnd = df_rnd[df_rnd.columns.drop(list(df_rnd.filter(regex='C2RL')))]

df_rnd['Real']=0

df = pd.concat([df_rnd, df])

df = df.loc[:,df.apply(pd.Series.nunique) != 1]

groups =   df["Network"]
cv_outer = GroupKFold(n_splits = 5)
y = df['Real']
X = df.drop(df.iloc[:, 0:10], axis=1)
X = X[X.columns.drop(list(X.filter(regex='Real')))]
recalls_1 = list()
recalls_0 = list()
precisions_1 = list()
precisions_0 = list()
accuracies= list()
outer_srcc = list()
outer_features_count = list()
outer_features = list()
outer_features_importance = list()
outer_pc_count = list()
outer_fold = 1
for train_inds, test_inds in cv_outer.split(X, y, groups):

    results_path = "../data/results/"+arch+"_CLASSIF"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = X.iloc[train_inds], X.iloc[test_inds], y.iloc[train_inds], y.iloc[test_inds]
    base_estimator = RandomForestClassifier(random_state=0)
    scaler = MinMaxScaler()
    selector = VarianceThreshold()
    kbest = SelectPercentile()
    pipeline = Pipeline([('scaler', scaler),('kbest', kbest),('be', base_estimator)])
    param_grid = {'kbest__score_func': [f_classif],'kbest__percentile': list(range(10,101,10))}
    
    grid = GridSearchCV(pipeline, param_grid, cv=GroupKFold(n_splits=5),refit=True, n_jobs = 5).fit(X_train, y_train,groups.iloc[train_inds])
    
    my_classifier = grid.best_estimator_
    y_pred = my_classifier.predict(X_test)
    
    recalls_1.append(recall_score(y_test, y_pred, pos_label=1))
    recalls_0.append(recall_score(y_test, y_pred, pos_label=0))
    precisions_1.append(precision_score(y_test, y_pred, pos_label=1))
    precisions_0.append(precision_score(y_test, y_pred, pos_label=0))
    accuracies.append(accuracy_score(y_test, y_pred))

        
    ##################################################
    ### Ranking of Feature Selection (SelectKBest) ###
    ##################################################
    
    # Get the indices sorted by most important to least important
    indices = np.argsort(my_classifier.named_steps["kbest"].scores_)[::-1]
    
    # To get your top x feature names
    topx = 10
    features = []
    for i in range(topx):
        features.insert(0,X_train.columns[indices[i]])

    features = [feature.replace('protcol', 'protocol') for feature in features] # fix spelling error
    features = [feature.replace('between', 'btw') for feature in features] 
    
    # Now plot
    f = plt.figure(figsize=(4,3.0))
    plt.barh(features, my_classifier.named_steps["kbest"].scores_[indices[range(topx)]][::-1],zorder=2, edgecolor='black',color='#0d52a8', align='center')

    plt.grid(zorder=0, linestyle="--", alpha=0.15)
    plt.yticks(size='12.0')
    plt.xticks(size='12.0')
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('ANOVA F-value',size='14.0')
    
    f.savefig(results_path+"/fold"+str(outer_fold)+"_SEL.pdf", bbox_inches='tight')
    ##########################################
    ### Ranking of ML Model (RandomForest) ###
    ##########################################
    
    # Identify the selected features first, as now some features are dropped and the index differs from original index (i.e., differs from X.columns)
    support = my_classifier.named_steps["kbest"].get_support()
    X_selected = X.columns[support]
    importances = my_classifier.named_steps["be"].feature_importances_
    
    # Now repeat the same as above
    indices = np.argsort(importances)[::-1]
    
    features = []
    for i in range(topx):
        features.insert(0,X_selected[indices[i]])

    f = plt.figure(figsize=(4,3.0))
    plt.barh(features, importances[indices[range(topx)]][::-1],zorder=2, edgecolor='black',color='#0d52a8', align='center')

    plt.grid(zorder=0, linestyle="--", alpha=0.15)
    plt.yticks(size='12.0')
    plt.xticks(size='12.0')
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('MDI',size='14.0')
    plt.show()
 
    f.savefig(results_path+"/fold"+str(outer_fold)+"_RF.pdf", bbox_inches='tight')

    outer_features_df = pd.DataFrame(outer_features)
    outer_features_df.to_csv(results_path+"/features.csv")
    np.savetxt(results_path+"/featurecount.csv", outer_features_count, delimiter=",")
    np.savetxt(results_path+"/accuracies.csv", accuracies, delimiter=",")
    np.savetxt(results_path+"/precisions_0.csv", precisions_0, delimiter=",")
    np.savetxt(results_path+"/recalls_0.csv", recalls_0, delimiter=",")
    np.savetxt(results_path+"/precisions_1.csv", precisions_1, delimiter=",")
    np.savetxt(results_path+"/recalls_1.csv", recalls_1, delimiter=",")

    
    outer_features_importance_df = pd.DataFrame(outer_features_importance)
    outer_features_importance_df.to_csv(results_path+"/featureimportances.csv")

    outer_fold = outer_fold + 1



