This repository contains slimmed-down code for the Context-Predictive Words unsupervised keyphrase extractor, developed to solve problems in Stance Clustering.

If you're interested in understanding the context for developing this keyphrase extractor, check out some of the documents in `relevant_papers`. If you're interested in using this keyphrase extractor, please reach out to me and I would be happy to assist.

The main file is in `code/CPW.py`.
In there is extensive documentation in that file describing how to use this extractor.

In order for the out-of-the-box code to run, you'll need to download the (large) Bitterlemons dataset, which can be found here: http://curtis.ml.cmu.edu/w/courses/index.php/Bitterlemons_dataset

Additionally, you'll need to do some data preprocessing. All the needed functions are in `code/utils.py`
 or `code/baselines.py`.
