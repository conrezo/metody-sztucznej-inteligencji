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
#czy powinniśmy to przepuścić przez wszystkie metryki i wszystkie clfs? mnóstwo przypadków...

evaluator_sudden = sl.evaluators.TestThenTrain(metrics)
evaluator_gradual = sl.evaluators.TestThenTrain(metrics)
evaluator_incremental = sl.evaluators.TestThenTrain(metrics)

""" próba podziału - ślepa uliczka
evaluator_sudden_UOB = sl.evaluators.TestThenTrain(metrics)
evaluator_sudden_OOB = sl.evaluators.TestThenTrain(metrics)
evaluator_sudden_OB = sl.evaluators.TestThenTrain(metrics)
evaluator_sudden_SEA = sl.evaluators.TestThenTrain(metrics)

evaluator_gradual_UOB = sl.evaluators.TestThenTrain(metrics)
evaluator_gradual_OOB = sl.evaluators.TestThenTrain(metrics)
evaluator_gradual_OB = sl.evaluators.TestThenTrain(metrics)
evaluator_gradual_SEA = sl.evaluators.TestThenTrain(metrics)

evaluator_incremental_UOB = sl.evaluators.TestThenTrain(metrics)
evaluator_incremental_OOB = sl.evaluators.TestThenTrain(metrics)
evaluator_incremental_OB = sl.evaluators.TestThenTrain(metrics)
evaluator_incremental_SEA = sl.evaluators.TestThenTrain(metrics)
"""

#run evaluators
evaluator_sudden.process(stream_sudden, clfs.values())
evaluator_gradual.process(stream_gradual, clfs.values())
evaluator_incremental.process(stream_incremental, clfs.values())

""" 
evaluator_sudden_UOB.process(stream_sudden, clfs.get('UOB'))
evaluator_sudden_OOB.process(stream_sudden, clfs.get('OOB'))
evaluator_sudden_OB.process(stream_sudden, clfs.get('OB'))
evaluator_sudden_SEA.process(stream_sudden, clfs.get('SEA'))

evaluator_gradual_UOB.process(stream_gradual, clfs.get('UOB'))
evaluator_gradual_OOB.process(stream_gradual, clfs.get('OOB'))
evaluator_gradual_OB.process(stream_gradual, clfs.get('OB'))
evaluator_gradual_SEA.process(stream_gradual, clfs.get('SEA'))

evaluator_incremental_UOB.process(stream_incremental, clfs.get('UOB')) 
evaluator_incremental_OOB.process(stream_incremental, clfs.get('OOB')) 
evaluator_incremental_OB.process(stream_incremental, clfs.get('OB')) 
evaluator_incremental_SEA.process(stream_incremental, clfs.get('SEA')) 
"""

#print scores results
#print(evaluator_sudden.scores)

#saving results (evaluator scores) to file
def save_to_file(evaluator, drift_name:str):
    np.save('results_' + drift_name, evaluator.scores)

save_to_file(evaluator_sudden, "sudden")
save_to_file(evaluator_gradual, "gradual")
save_to_file(evaluator_incremental, "incremental")



################ DATA ANALYSIS ############################################################

#reading_result_from_file 
scores_sudden = np.load('results_sudden.npy')
print("\nScores (sudden):\n", scores_sudden)

scores_gradual = np.load('results_gradual.npy')
print("\nScores (gradual):\n", scores_gradual)

scores_incremental = np.load('results_incremental.npy')
print("\nScores (incremental):\n", scores_incremental)


#mean scores 
mean_scores_sudden = np.mean(scores_sudden) # jak dobrać axis?
print("\nMean scores (sudden):\n", mean_scores_sudden)

mean_scores_gradual = np.mean(scores_gradual)
print("\nMean scores (gradual):\n", mean_scores_gradual)

mean_scores_incremental = np.mean(scores_incremental)
print("\nMean scores (incremental):\n", mean_scores_incremental)


#std scores
std_scores_sudden = np.std(scores_sudden)
print("\nStd scores (sudden):\n", std_scores_sudden)

std_scores_gradual = np.std(scores_gradual)
print("\nStd scores (gradual):\n", std_scores_gradual)

std_scores_incremental = np.std(scores_incremental)
print("\nStd scores (incremental):\n", std_scores_incremental)



############## PRESENTING RESULTS ######################################################

def show_results(mean, std): 
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean, std)) #według przykładu - (clf_name, mean[clf_id], std[clf_id]), wtedy błąd indeksu

print("sudden")
show_results(mean_scores_sudden, std_scores_sudden)
print("gradual")
show_results(mean_scores_gradual, std_scores_gradual)
print("incremental")
show_results(mean_scores_incremental, std_scores_incremental)



"""
############# STATISTICAL ANALYSIS #################################################################

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

#creating t_statistic and p_value matrices
def get_t_statistics_p_value(scores):
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

print("Scores sudden \n")
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