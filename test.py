from strlearn.ensembles import OOB, UOB, SEA, OnlineBagging
from sklearn.naive_bayes import GaussianNB
import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from tabulate import tabulate

#classificators
clfs = {   
    'UOB': UOB(base_estimator=GaussianNB(), n_estimators=5),
    'OOB': OOB(base_estimator=GaussianNB(), n_estimators=5),
    'OB': OnlineBagging(base_estimator=GaussianNB(), n_estimators=5),
    'SEA': SEA(base_estimator=GaussianNB(), n_estimators=5)
}


#metrics
metrics = [sl.metrics.f1_score,
           sl.metrics.geometric_mean_score_1]

#metrics names
metrics_names = ["F1 score",
                 "G-mean"]



########### GENERATING STREAMS #######################################################

##generate sudden drift
stream_sudden = sl.streams.StreamGenerator(n_chunks=200,
                                            chunk_size=500,
                                            n_classes=2,
                                            n_drifts=1,
                                            n_features=10,
                                            random_state=10)

#generatre gradual drift
stream_gradual = sl.streams.StreamGenerator(n_chunks=200,
                                            chunk_size=500,
                                            n_classes=2,
                                            n_drifts=1,
                                            n_features=10,
                                            random_state=10,
                                            concept_sigmoid_spacing=5)

#generate incremental drift
stream_incremental = sl.streams.StreamGenerator(n_chunks=200,
                                                chunk_size=500,
                                                n_classes=2,
                                                n_drifts=1,
                                                n_features=10,
                                                random_state=10,
                                                concept_sigmoid_spacing=5,
                                                incremental=True)


#evaluator initialization
evaluator_sudden = sl.evaluators.TestThenTrain(metrics)
evaluator_gradual = sl.evaluators.TestThenTrain(metrics)
evaluator_incremental = sl.evaluators.TestThenTrain(metrics)

#run evaluators
evaluator_sudden.process(stream_sudden, clfs.values())
evaluator_gradual.process(stream_gradual, clfs.values())
evaluator_incremental.process(stream_incremental, clfs.values())

#print scores results
#print(evaluator_sudden.scores)

#saving results (evaluator scores) to file
def save_to_file(evaluator, drift_name:str):
    np.save('results_' + drift_name, evaluator.scores)

save_to_file(evaluator_sudden, "sudden")
save_to_file(evaluator_gradual, "gradual")
save_to_file(evaluator_incremental, "incremental")



################ DATA ANALYSIS ######################################################

#reading_result_from_file 
scores_sudden = np.load('results_sudden.npy')
print("\nScores (sudden):\n", scores_sudden)

scores_gradual = np.load('results_gradual.npy')
print("\nScores (gradual):\n", scores_gradual)

scores_incremental = np.load('results_incremental.npy')
print("\nScores (incremental):\n", scores_incremental)


#mean scores 
mean_sudden = np.mean(scores_sudden, axis=1) 
print("\n\nMean (sudden):\n", mean_sudden)

mean_gradual = np.mean(scores_gradual, axis=1)
print("\nMean (gradual):\n", mean_gradual)

mean_incremental = np.mean(scores_incremental, axis=1)
print("\nMean (incremental):\n", mean_incremental)


#std scores
std_sudden = np.std(scores_sudden, axis=1)
print("\n\nStd (sudden):\n", std_sudden)

std_gradual = np.std(scores_gradual, axis=1)
print("\nStd (gradual):\n", std_gradual)

std_incremental = np.std(scores_incremental, axis=1)
print("\nStd (incremental):\n", std_incremental)


#preparing mean and std to presenting results - is it necessary? 
#create lists for  values
mean_sudden_list = []
mean_gradual_list = []
mean_incremental_list = []

std_sudden_list = []
std_gradual_list = []
std_incremental_list = []

#count avg mean and avg std for specific clfs (one per row)
def calculate_avg(type, newlist, eid: int):
    avg=((type[eid][0]+type[eid][1])/2)
    newlist.append(avg)
    return avg

for clf_id, clf_name in enumerate(clfs):
    calculate_avg(mean_sudden, mean_sudden_list, clf_id)
    calculate_avg(mean_gradual, mean_gradual_list, clf_id)
    calculate_avg(mean_incremental, mean_incremental_list, clf_id)
    calculate_avg(std_sudden, std_sudden_list, clf_id)
    calculate_avg(std_gradual, std_gradual_list, clf_id)
    calculate_avg(std_incremental, std_incremental_list, clf_id)



############## PRESENTING RESULTS ######################################################

def show_results(mean, std): 
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id])) 

print("\n\nResults (sudden):")
show_results(mean_sudden_list, std_sudden_list)
print("\nResults (gradual):")
show_results(mean_gradual_list, std_gradual_list)
print("\nResults (incremental):")
show_results(mean_incremental_list, std_incremental_list)


"""
############# STATISTICAL ANALYSIS ######################################################

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

#creating t_statistic and p_value matrices
def get_t_statistics_p_value(scores):
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

print("\nScores sudden \n")
get_t_statistics_p_value(scores_sudden)
print("Scores gradual \n")
get_t_statistics_p_value(scores_gradual)
print("Scores incremental \n")
get_t_statistics_p_value(scores_incremental)

headers = ["GNB", "kNN", "CART"]
names_column = np.array([["GNB"], ["kNN"], ["CART"]])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


#advantage matrics
advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)


#significance table
significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)

#final results
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
"""