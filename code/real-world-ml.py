# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:58:37 2021

@author: katha
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectPercentile
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
from scipy import stats
import itertools
import pickle
from pathlib import Path
from scipy.cluster import hierarchy
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin

# adapted from https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html, so: Copyright (c) 2007-2021 The scikit-learn developers. All rights reserved.
class MulticollinearityFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.selected_features = []
        
    def fit(self, X, y = None ):
        X_df = pd.DataFrame(X)
        corr = stats.spearmanr(X_df).correlation
        corr_linkage = hierarchy.ward(corr)

        cluster_ids = hierarchy.fcluster(corr_linkage, self.threshold, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)
        self.selected_features = list(X_df.columns[[v[0] for v in cluster_id_to_feature_ids.values()]])       
        return self 
    
    def transform(self, X, y = None ):
        X_df = pd.DataFrame(X)
        X_df = X_df[self.selected_features]
        return X_df
    
archs = ["HF"]
#archs = ["KN"]
pipes = ["cluster_kbest","kbest_cluster","pca","pca_kbest","kbest_pca"]
scenarios = ["N"]
#models = ["DT","AB","RF"]
models = ["DT"]
metrics = ["Mean_RTT","Max_RTT", "Mean_SyncTraffic", "Max_SyncTraffic", "Mean_ControlPlaneTraffic", "Max_ControlPlaneTraffic"]

z = itertools.product(archs, scenarios, models, pipes, metrics)
for arch, scenario, model, pipe, metric in z:
    
    if (arch == "KN"):
        df = pd.read_csv('../data/data_KN.csv', sep=";").fillna(value=-1)
    if (arch == "HF"):
        df = pd.read_csv('../data/data_HF.csv', sep=";").fillna(value=-1)
        df = df[df.columns.drop(list(df.filter(regex='C2RL')))]
       
    results_path = "../data/results/"+arch+"_"+model+"_"+ scenario+"_"+pipe
    Path(results_path).mkdir(parents=True, exist_ok=True)
     
    
    # Group the same network from different years also together... otherwise it may be unfair, cause they stay very similar
    if (scenario == "NC"):
        df['Network'].replace(to_replace='[0-9]*', value='',inplace=True,regex=True) 
        df['Network'].replace(to_replace='KentmanFeb|KentmanJan', value='Kentman',inplace=True,regex=True) 
        groups =  df["Network"] + "-" +df["Configuration"].astype(str)
        
    if (scenario == "N"):

        df['Network'].replace(to_replace='[0-9]*', value='',inplace=True,regex=True) 
        df['Network'].replace(to_replace='KentmanFeb|KentmanJan', value='Kentman',inplace=True,regex=True) 
        groups =  df["Network"]

    y = df[metric]
    X = df.drop(df.iloc[:, 0:10], axis=1)
    
    cv_outer = GroupKFold(n_splits = 5)
   
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
    for train_inds, test_inds in cv_outer.split(X, y, groups):
    
        X_train, X_test, y_train, y_test = X.iloc[train_inds], X.iloc[test_inds], y.iloc[train_inds], y.iloc[test_inds]
        
        
        if (model == "DT"):
            base_estimator = DecisionTreeRegressor(random_state=0)
        if (model == "RF"):
            base_estimator = RandomForestRegressor(random_state=0)
        if (model == "AB"):
            ab_estimator = DecisionTreeRegressor(random_state=0)
            base_estimator = AdaBoostRegressor(ab_estimator,random_state=0)
        
        if (pipe == "baseline"):
            scaler = MinMaxScaler()
            selector = VarianceThreshold()
            kbest = SelectPercentile()
            pipeline = Pipeline([('selector', selector),('scaler', scaler),('kbest', kbest),('be', base_estimator)])
            param_grid = {'kbest__score_func': [f_regression],'kbest__percentile': list(range(10,101,10))}
            
        if (pipe == "pca"):
            scaler = StandardScaler()
            selector = VarianceThreshold()
            pca = PCA(svd_solver='full',random_state=0)
            pipeline = Pipeline([('selector', selector),('scaler', scaler),('pca', pca),('be', base_estimator)])
            param_grid = {'pca__n_components': [x / 10 for x in range(1, 10, 1)] + [None]} # None equals all PCs
            
        if (pipe == "pca_kbest"):
            scaler = StandardScaler()
            selector = VarianceThreshold()
            pca = PCA(svd_solver='full',random_state=0)
            kbest = SelectPercentile()
            pipeline = Pipeline([('selector', selector),('scaler', scaler),('pca', pca),('kbest', kbest), ('be', base_estimator)])
            param_grid = {'kbest__score_func': [f_regression],'kbest__percentile': list(range(10,101,10))}
            
        if (pipe == "kbest_pca"):
            scaler = StandardScaler()
            selector = VarianceThreshold()
            pca = PCA(svd_solver='full',random_state=0)
            kbest = SelectPercentile()
            pipeline = Pipeline([('selector', selector),('scaler', scaler),('kbest', kbest),('pca', pca),('be', base_estimator)])
            param_grid = {'kbest__score_func': [f_regression],'kbest__percentile': list(range(10,101,10))}
             
        if (pipe == "cluster_kbest"):
           scaler = MinMaxScaler()
           selector = VarianceThreshold()
           kbest = SelectPercentile()
           corr_filter = MulticollinearityFilter(5)         
           pipeline = Pipeline([('selector', selector),('scaler', scaler),('corr_filter', corr_filter),('kbest', kbest), ('be', base_estimator)])
           param_grid = {'kbest__score_func': [f_regression],'kbest__percentile': list(range(10,101,10))}
            
        if (pipe == "kbest_cluster"):
           scaler = MinMaxScaler()
           selector = VarianceThreshold()
           kbest = SelectPercentile()
           corr_filter = MulticollinearityFilter(5)
           pipeline = Pipeline([('selector', selector),('scaler', scaler),('kbest', kbest),('corr_filter', corr_filter),('be', base_estimator)])
           param_grid = {'kbest__score_func': [f_regression],'kbest__percentile': list(range(10,101,10))}
           

        # Cross Validation
        grid = GridSearchCV(pipeline, param_grid, cv=GroupKFold(n_splits=5),refit=True, n_jobs = 5).fit(X_train, y_train,groups.iloc[train_inds])
        
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
        
        if (pipe == "pca_kbest"):
            outer_features_count.append(len(my_regressor.named_steps["kbest"].get_support(indices = True)))
        if (pipe == "kbest_pca"):
            outer_pc_count.append(my_regressor.named_steps["pca"].n_components_)
        if (pipe == "kbest_cluster"):          
            outer_features_count.append(len(my_regressor.named_steps['corr_filter'].selected_features))
        if (pipe == "pca"):
            outer_pc_count.append(my_regressor.named_steps["pca"].n_components_)
        if (pipe == "baseline"):
            outer_features_count.append(len(my_regressor.named_steps["kbest"].get_support(indices = True)))
            
            featureIndices_selector = my_regressor.named_steps["selector"].get_support(indices = True)
            featureIndices_kbest = my_regressor.named_steps["kbest"].get_support(indices = True)
            retained_df = X_train.iloc[:,featureIndices_selector].iloc[:,featureIndices_kbest]
            
            outer_features.append(retained_df.columns)          
            
            importances = my_regressor.named_steps['kbest'].scores_[featureIndices_kbest]
            outer_features_importance.append(importances)
        if (pipe == "cluster_kbest"):
            outer_features_count.append(len(my_regressor.named_steps["kbest"].get_support(indices = True)))
            
            featureIndices_kbest = my_regressor.named_steps["kbest"].get_support(indices = True)
            importances = my_regressor.named_steps['kbest'].scores_[featureIndices_kbest]
            outer_features_importance.append(importances)
            
            featureIndices_selector = my_regressor.named_steps["selector"].get_support(indices = True)
            retained_df = X_train.iloc[:,featureIndices_selector]
            
            featureIndices_filter = my_regressor.named_steps['corr_filter'].selected_features
            
            sel_list = list()
            for index in featureIndices_kbest.tolist():      
                sel_list.append(retained_df.columns[featureIndices_filter[index]])
            outer_features.append(sel_list)
                    
        # Uncomment if backup of ML model and train/test targets is wanted        
        # Path(results_path+"/backup").mkdir(parents=True, exist_ok=True)
        # filename = results_path+"/backup"+"/model_"+  str(outer_fold) +"_" +metric+".sav"
        # np.savetxt(results_path+"/backup""/ytest_" + str(outer_fold) +"_"+metric+".csv", y_test, delimiter=",")
        # np.savetxt(results_path+"/backup""/ypred_" + str(outer_fold) +"_"+metric+".csv", y_pred, delimiter=",")
        # pickle.dump(my_regressor, open(filename, 'wb'))
        
        outer_fold = outer_fold + 1
    
    np.savetxt(results_path+"/mape_"+metric+".csv", outer_mape, delimiter=",")
    np.savetxt(results_path+"/mae_"+metric+".csv", outer_mae, delimiter=",")
    np.savetxt(results_path+"/rmse_"+metric+".csv", outer_rmse, delimiter=",")
    np.savetxt(results_path+"/nmae_"+metric+".csv", outer_nmae, delimiter=",")
    np.savetxt(results_path+"/srcc_"+metric+".csv", outer_srcc, delimiter=",")
    np.savetxt(results_path+"/pcc_"+metric+".csv", outer_pcc, delimiter=",")
        
    if (pipe == "pca_kbest" or pipe == "kbest_cluster"): 
        np.savetxt(results_path+"/featurecount_"+metric+".csv", outer_features_count, delimiter=",")
    if (pipe == "baseline" or pipe == "cluster_kbest"): 
        outer_features_df = pd.DataFrame(outer_features)
        outer_features_df.to_csv(results_path+"/features_"+metric+".csv")
        np.savetxt(results_path+"/featurecount_"+metric+".csv", outer_features_count, delimiter=",")
        
        outer_features_importance_df = pd.DataFrame(outer_features_importance)
        outer_features_importance_df.to_csv(results_path+"/featureimportances_"+metric+".csv")
    if (pipe == "pca" or pipe == "kbest_pca"):
        np.savetxt(results_path+"/pccount_"+metric+".csv", outer_pc_count, delimiter=",")