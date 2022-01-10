# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:33:07 2021

@author: katha
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
from matplotlib import ticker
import re
import matplotlib as mpl

scenario = "NC"
model = "RF"
metric = "Mean_ControlPlaneTraffic"
#metrics = ["RTT","SyncTraffic","ControlPlaneTraffic"]
#metrics = ["Mean_ControlPlaneTraffic"]
pipelines = ["","_kbest_cluster","_cluster_kbest", "_pca","_pca_kbest","_kbest_pca"]
#pipelines = [""]
colors = ["#4473c4","#ffe699","#ffc000","#a9d18e","#70ad47","#548235"]
fills = ['x','x','x','x','x','x']

#names_metric = ["ControlPlaneTraffic", "SyncTraffic", "RTT"]
names_architecture = ["hf"]
names_combined = itertools.product(names_architecture,pipelines)
results_fixed = pd.DataFrame(columns=["metric", "model","scenario","pipeline","featurecount"])
for arch,pipeline in names_combined:
   

    if pipeline == "":
          result = np.loadtxt("../data/results_fixed/"+arch+"_"+model+"_"+ scenario+pipeline+"/featurecount_"+metric+".csv")
    elif pipeline == "_cluster_kbest":
        result = np.loadtxt("../data/results_fixed/"+arch+"_"+model+"_"+ scenario+pipeline+"/featurecount_"+metric+".csv")
    elif pipeline == "_kbest_cluster" :
         result = np.loadtxt("../data/results_fixed/"+arch+"_"+model+"_"+ scenario+pipeline+"/featurecount_"+metric+".csv")
    elif pipeline == "_pca_kbest":
         result = np.loadtxt("../data/results_fixed/"+arch+"_"+model+"_"+ scenario+pipeline+"/featurecount_"+metric+".csv")
    elif pipeline == "_kbest_pca":
        result = np.loadtxt("../data/results_fixed/"+arch+"_"+model+"_"+ scenario+pipeline+"/featurecount_"+metric+".csv")
    else:
        result = np.loadtxt("../data/results_fixed/"+arch+"_"+model+"_"+ scenario+pipeline+"/pccount_"+metric+".csv")
    for subresult in result:
        if pipeline == "":
            pipeline = "base"
        results_fixed = results_fixed.append({"metric" : metric, "model" : model, "scenario" : scenario, "pipeline" : pipeline.replace("_", " "), "featurecount" : subresult}, ignore_index=True)
