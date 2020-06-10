from strlearn.ensembles import OOB, UOB, SEA, OnlineBagging
from sklearn.naive_bayes import GaussianNB

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

from scipy.stats import rankdata
from scipy.stats import ranksums


#classificators
clfs = {   
    'UOB': UOB(base_estimator=GaussianNB(), n_estimators=5),
    'OOB': OOB(base_estimator=GaussianNB(), n_estimators=5),
    'OB' : OnlineBagging(base_estimator=GaussianNB(), n_estimators=5),
    'SEA': SEA(base_estimator=GaussianNB(), n_estimators=5)
}

#our_data_sets
datasets = ['australian', 'breastcan', 'cryotherapy','diabetes','german','liver','heart','ionosphere', 'monkone', 'wisconsin']

#experiment
n_datasets = len(datasets)    #length of dict
n_splits = 5                  #number of splits
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

#reading_data_input 
for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

#saving_results_to_file
np.save('results', scores)


### DATA ANALYSIS ###

#reading_result_from_file 
scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

#mean scores 
mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

#ranks
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

#mean ranks
mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)