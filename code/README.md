# Source Code Overview

*real-world-ml.py* is the source code for the ML-based evaluation of real-world WANs for all pipelines

*real-world-table-eval.py* aggregates the results to show the results seen in the paper according to MAE

*synth-ml.py* is the source for the RF-based evaluation for the baseline pipeline

*synth-table-eval.py* aggregates the results to show the results seen in the paper, both the synthetic-based RF and the real-world-based RF

*dendro.py* is a small script to visualize the clustering and chosen threshold for it (here threshold is 5, also see *dendro.pdf*)

*top-features.py* is a small script to print out the top 30 features for an architecture and metrics for the baseline pipeline

*classify-real-vs-synth.py* builds an RF to classify between real and synthetic networks