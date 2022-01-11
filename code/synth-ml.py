# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:58:37 2021

@author: katha
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
import itertools
from pathlib import Path
import pickle

arch = "HF"
if (arch == "KN"):
    df = pd.read_csv('../data/data_KN.csv', sep=";").fillna(value=-1)
if (arch == "HF"):
    df = pd.read_csv('../data/data_HF.csv', sep=";").fillna(value=-1)
    df = df[df.columns.drop(list(df.filter(regex='C2RL')))]


if (arch == "KN"):
    df_rnd = pd.read_csv('../data/data_rnd_KN.csv', sep=";").fillna(value=-1)
if (arch == "HF"):
    df_rnd = pd.read_csv('../data/data_rnd_HF.csv', sep=";").fillna(value=-1)
    df_rnd = df_rnd[df_rnd.columns.drop(list(df_rnd.filter(regex='C2RL')))]


pipe = "baseline"
scenarios = ["N"]
models = ["RF"]
metrics = ["Mean_RTT", "Max_RTT", "Mean_SyncTraffic", "Max_SyncTraffic", "Mean_ControlPlaneTraffic", "Max_ControlPlaneTraffic"]

z = itertools.product(scenarios, models, metrics)
for scenario, model, metric in z:
    y_test = df[metric]
    X_test = df.drop(df.iloc[:, 0:10], axis=1)
   
    y_train = df_rnd[metric]
    X_train = df_rnd.drop(df.iloc[:, 0:10], axis=1)
   
    c1 = X_test.columns
    c2 = X_train.columns

    results_path = "../data/results/"+arch+"_RND_"+model+"_"+ scenario+"_"+pipe
    Path(results_path).mkdir(parents=True, exist_ok=True)
    if (scenario == "NC"):
        groups =   df_rnd["Network"] + "-" +df_rnd["Configuration"].astype(str)
        
    if (scenario == "N"):
        groups =   df_rnd["Network"]
        
    # enumerate splits
    outer_mape = list()
    outer_mae = list()
    outer_rmse = list()
    outer_nmae = list()
    outer_pcc= list()
    outer_srcc = list()
    outer_features_count = list()
    outer_features = list()
    outer_features_importance = list()
    outer_pc_count = list()
    outer_fold = 1

    scaler = MinMaxScaler()
    selector = VarianceThreshold()
    kbest = SelectPercentile()

    if (model == "DT"):
        base_estimator = DecisionTreeRegressor(random_state=0)
    if (model == "RF"):
        base_estimator = RandomForestRegressor(random_state=0)
    if (model == "LR"):
        base_estimator = LinearRegression()
    if (model == "AB"):
        ab_estimator = DecisionTreeRegressor(random_state=0)
        base_estimator = AdaBoostRegressor(ab_estimator)
    
    pipeline = Pipeline([('selector',selector),('scaler', scaler),('kbest', kbest), ('be', base_estimator)])
    
    # Parameter Tuning
    param_grid = {'kbest__score_func': [f_regression],'kbest__percentile': list(range(10,101,10))}
    
    # Cross Validation
    grid = GridSearchCV(pipeline, param_grid, cv=GroupKFold(n_splits=5),refit=True,verbose=5, n_jobs = 5).fit(X_train, y_train,groups)
    

    my_regressor = grid.best_estimator_
    y_pred = my_regressor.predict(X_test)
    
    error = y_test - y_pred
    outer_mae.append(np.mean(abs(error)))
    outer_rmse.append(np.sqrt(np.mean(np.square(error))))
    outer_nmae.append(np.mean(abs(error))/np.mean(y_test))
    
    mape = np.mean(abs(error)/abs(y_test))
    outer_mape.append(mape)
    
    outer_pcc.append(stats.pearsonr(y_pred, y_test)[0])
    outer_srcc.append(stats.spearmanr(y_pred, y_test)[0])

    outer_features_count.append(len(my_regressor.named_steps["kbest"].get_support(indices = True)))
    
    featureIndices_selector = my_regressor.named_steps["selector"].get_support(indices = True)
    featureIndices_kbest = my_regressor.named_steps["kbest"].get_support(indices = True)
    retained_df = X_train.iloc[:,featureIndices_selector].iloc[:,featureIndices_kbest]
    
    outer_features.append(retained_df.columns)          
    
    importances = my_regressor.named_steps['kbest'].scores_[featureIndices_kbest]
    outer_features_importance.append(importances)
    
    # Uncomment if backup of ML model and train/test targets is wanted    
    # Path(results_path+"/backup").mkdir(parents=True, exist_ok=True)
    # filename = results_path+"/backup"+"/model_"+  str(outer_fold) +"_" +metric+".sav"
    # np.savetxt(results_path+"/backup""/ytest_" + str(outer_fold) +"_"+metric+".csv", y_test, delimiter=",")
    # np.savetxt(results_path+"/backup""/ypred_" + str(outer_fold) +"_"+metric+".csv", y_pred, delimiter=",")
    # pickle.dump(my_regressor, open(filename, 'wb'))
    
    np.savetxt(results_path+"/mape_"+metric+".csv", outer_mape, delimiter=",")
    np.savetxt(results_path+"/mae_"+metric+".csv", outer_mae, delimiter=",")
    np.savetxt(results_path+"/rmse_"+metric+".csv", outer_rmse, delimiter=",")
    np.savetxt(results_path+"/nmae_"+metric+".csv", outer_nmae, delimiter=",")
    np.savetxt(results_path+"/srcc_"+metric+".csv", outer_srcc, delimiter=",")
    np.savetxt(results_path+"/pcc_"+metric+".csv", outer_pcc, delimiter=",")
    
    outer_fold = outer_fold + 1
    
    outer_features_importance_df = pd.DataFrame(outer_features_importance)
    
    outer_features_df = pd.DataFrame(outer_features)
    outer_features_df.to_csv(results_path+"/features_"+metric+".csv")
    np.savetxt(results_path+"/featurecount_"+metric+".csv", outer_features_count, delimiter=",")
        
    outer_features_importance_df = pd.DataFrame(outer_features_importance)
    outer_features_importance_df.to_csv(results_path+"/featureimportances_"+metric+".csv")