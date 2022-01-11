# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 02:46:01 2021

@author: katha
"""

import pandas as pd
import numpy as np
import itertools


scenarios = ["N"]
models = ["RF"]
metrics = ["Mean_RTT", "Max_RTT", "Mean_SyncTraffic", "Max_SyncTraffic", "Mean_ControlPlaneTraffic", "Max_ControlPlaneTraffic"]
pipelines = ["_baseline"]

arch = "HF"

results_mae = pd.DataFrame(columns=["metric", "model","scenario","pipeline","mae"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_RND_"+model+"_"+ scenario+pipeline+"/mae_"+metric+".csv")
    if metric == "Max_RTT" or metric == "Mean_RTT":
        result = result * 1000
    results_mae = results_mae.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "mae" : result}, ignore_index=True)

results_mape = pd.DataFrame(columns=["metric", "model","scenario","pipeline","mape"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_RND_"+model+"_"+ scenario+pipeline+"/mape_"+metric+".csv")
    results_mape = results_mape.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "mape" : result}, ignore_index=True)
        
results_nmae = pd.DataFrame(columns=["metric", "model","scenario","pipeline","nmae"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_RND_"+model+"_"+ scenario+pipeline+"/nmae_"+metric+".csv")
    results_nmae = results_nmae.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "nmae" : result}, ignore_index=True)

results_rmse= pd.DataFrame(columns=["metric", "model","scenario","pipeline","rmse"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_RND_"+model+"_"+ scenario+pipeline+"/rmse_"+metric+".csv")
    if metric == "Max_RTT" or metric == "Mean_RTT":
        result = result * 1000
    results_rmse = results_rmse.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "rmse" : result}, ignore_index=True)

results_pcc= pd.DataFrame(columns=["metric", "model","scenario","pipeline","pcc"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_RND_"+model+"_"+ scenario+pipeline+"/pcc_"+metric+".csv")
    results_pcc = results_pcc.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "pcc" : result}, ignore_index=True)

results_srcc= pd.DataFrame(columns=["metric", "model","scenario","pipeline","srcc"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_RND_"+model+"_"+ scenario+pipeline+"/srcc_"+metric+".csv")
    results_srcc = results_srcc.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "srcc" : result}, ignore_index=True)

results_agg3 = results_rmse.merge(results_mape, how='left', on=["metric","model","scenario","pipeline"])
results_agg2 = results_agg3.merge(results_nmae, how='left', on=["metric","model","scenario","pipeline"]) 
results_agg1 = results_agg2.merge(results_pcc, how='left', on=["metric","model","scenario","pipeline"])
results_agg0 = results_agg1.merge(results_srcc, how='left', on=["metric","model","scenario","pipeline"])
results_synth = results_mae.merge(results_agg0, how='left', on=["metric","model","scenario","pipeline"])
  
print(results_synth)

results_mae = pd.DataFrame(columns=["metric", "model","scenario","pipeline","mae"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_"+model+"_"+ scenario+pipeline+"/mae_"+metric+".csv")
    for subresult in result:
        if metric == "Max_RTT" or metric == "Mean_RTT":
            subresult = subresult * 1000
        results_mae = results_mae.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "mae" : subresult}, ignore_index=True)

results_mape = pd.DataFrame(columns=["metric", "model","scenario","pipeline","mape"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_"+model+"_"+ scenario+pipeline+"/mape_"+metric+".csv")
    for subresult in result:
        results_mape = results_mape.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "mape" : subresult}, ignore_index=True)
        
results_nmae = pd.DataFrame(columns=["metric", "model","scenario","pipeline","nmae"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_"+model+"_"+ scenario+pipeline+"/nmae_"+metric+".csv")
    for subresult in result:
        results_nmae = results_nmae.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "nmae" : subresult}, ignore_index=True)

results_rmse= pd.DataFrame(columns=["metric", "model","scenario","pipeline","rmse"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_"+model+"_"+ scenario+pipeline+"/rmse_"+metric+".csv")
    for subresult in result:
        if metric == "Max_RTT" or metric == "Mean_RTT":
            subresult = subresult * 1000
        results_rmse = results_rmse.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "rmse" : subresult}, ignore_index=True)

results_pcc= pd.DataFrame(columns=["metric", "model","scenario","pipeline","pcc"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_"+model+"_"+ scenario+pipeline+"/pcc_"+metric+".csv")
    for subresult in result:
        results_pcc = results_pcc.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "pcc" : subresult}, ignore_index=True)

results_srcc= pd.DataFrame(columns=["metric", "model","scenario","pipeline","srcc"])
z = itertools.product(models,scenarios, metrics, pipelines)
for model,scenario, metric, pipeline in z:
    result = np.loadtxt("../data/results/"+arch+"_"+model+"_"+ scenario+pipeline+"/srcc_"+metric+".csv")
    for subresult in result:
        results_srcc = results_srcc.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "srcc" : subresult}, ignore_index=True)

# Average over five outer folds is needed here, we do not have an outer CV for the synthetic networks though
results_agg_nmae = results_nmae.groupby(["metric","model","scenario","pipeline"]).mean()
results_agg_mape = results_mape.groupby(["metric","model","scenario","pipeline"]).mean()
results_agg_rmse = results_rmse.groupby(["metric","model","scenario","pipeline"]).mean()
results_agg_pcc = results_pcc.groupby(["metric","model","scenario","pipeline"]).mean()
results_agg_srcc = results_srcc.groupby(["metric","model","scenario","pipeline"]).mean()
results_agg_mae = results_mae.groupby(["metric","model","scenario","pipeline"]).mean()

results_agg3 = results_agg_rmse.merge(results_agg_mape, how='left', on=["metric","model","scenario","pipeline"])
results_agg2 = results_agg3.merge(results_agg_nmae, how='left', on=["metric","model","scenario","pipeline"]) 
results_agg1 = results_agg2.merge(results_agg_pcc, how='left', on=["metric","model","scenario","pipeline"])
results_agg0 = results_agg1.merge(results_agg_srcc, how='left', on=["metric","model","scenario","pipeline"])
results_real = results_agg_mae.merge(results_agg0, how='left', on=["metric","model","scenario","pipeline"])
  
print(results_real)