import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from strlearn.ensembles import OOB, UOB, SEA, OnlineBagging
from tabulate import tabulate

#classificators
clfs = {   
    'UOB': UOB(base_estimator=GaussianNB(), n_estimators=5),
    'OOB': OOB(base_estimator=GaussianNB(), n_estimators=5),
    'OB' : OnlineBagging(base_estimator=GaussianNB(), n_estimators=5),
    'SEA': SEA(base_estimator=GaussianNB(), n_estimators=5)
}

#our_data_sets
datasets = ['australian',
            'breastcan',
            'cryotherapy',
            'diabetes',
            'german',
            'liver',
            'heart',
            'ionosphere',
            'monkone',
            'wisconsin'
            ]

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

# w-statistic and p-value
alfa = 0.05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

# Advantage
advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

# Statistical significance
significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

# Statistical significance better
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
