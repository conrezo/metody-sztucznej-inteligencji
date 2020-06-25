import numpy as np
import strlearn as sl
from strlearn.ensembles import OOB, UOB, SEA, OnlineBagging
from sklearn.naive_bayes import GaussianNB
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
metrics = [sl.metrics.f1_score] 
#other: sl.metrics.geometric_mean_score_1

#metrics names
metrics_names = ["F1 score"] 
#other: "G-mean"


########### GENERATING STREAMS #######################################################
n_chunks_value = 10 #todo zminieć  na 200


def generating_streams(random_state_value):
    #generate sudden drift
    stream_sudden = sl.streams.StreamGenerator(n_chunks=n_chunks_value,
                                               chunk_size=500,
                                               n_classes=2,
                                               n_drifts=1,
                                               n_features=10,
                                               random_state=random_state_value)
    evaluator_sudden = sl.evaluators.TestThenTrain(metrics)
    evaluator_sudden.process(stream_sudden, clfs.values())

    #generatre gradual drift
    stream_gradual = sl.streams.StreamGenerator(n_chunks=n_chunks_value,
                                                chunk_size=500,
                                                n_classes=2,
                                                n_drifts=1,
                                                n_features=10,
                                                random_state=random_state_value,
                                                concept_sigmoid_spacing=5)
    evaluator_gradual = sl.evaluators.TestThenTrain(metrics)
    evaluator_gradual.process(stream_gradual, clfs.values())

    #generate incremental drift
    stream_incremental = sl.streams.StreamGenerator(n_chunks=n_chunks_value,
                                                    chunk_size=500,
                                                    n_classes=2,
                                                    n_drifts=1,
                                                    n_features=10,
                                                    random_state=random_state_value,
                                                    concept_sigmoid_spacing=5,
                                                    incremental=True)

    evaluator_incremental = sl.evaluators.TestThenTrain(metrics)
    evaluator_incremental.process(stream_incremental, clfs.values())
    #evaluator initialization

    #run evaluators

    return evaluator_sudden.scores, evaluator_gradual.scores,  evaluator_incremental.scores


random_sate_list = [10, 1410, 21, 653, 1234, 190, 859, 329, 2137, 929]
#full_scores_sudden = [[[None for y in range(1)] for x in range(n_chunks_value-1)]]
#full_scores_gradual = [[[None for y in range(1)] for x in range(n_chunks_value-1)]]
#full_scores_incremental = [[[None for y in range(1)] for x in range(n_chunks_value-1)]]

full_scores_sudden = []
full_scores_gradual = []
full_scores_incremental = []



i = 0
for random_state in random_sate_list:
    scores_sudden, scores_gradual, scores_incremental = generating_streams(random_state)
    if i == 0:
        full_scores_sudden = scores_sudden
        full_scores_gradual = scores_gradual
        full_scores_incremental = scores_incremental
        i = 1
    else:
        # Sudden:
        arr = np.append(full_scores_sudden, scores_sudden, axis=0)
        full_scores_sudden = arr

        # Gradual:
        arr = np.append(full_scores_gradual, scores_gradual, axis=0)
        full_scores_gradual = arr

        # Incremental:
        arr = np.append(full_scores_incremental, scores_incremental, axis=0)
        full_scores_incremental = arr
        i = 0


#saving results (evaluator scores) to file
def save_to_file(full_scores, drift_name: str):
    np.save('results_' + drift_name, full_scores)

save_to_file(full_scores_sudden, "sudden")
save_to_file(full_scores_gradual, "gradual")
save_to_file(full_scores_incremental, "incremental")

################ DATA ANALYSIS ######################################################

#reading_result_from_file 
scores_sudden = np.load('results_sudden.npy', allow_pickle=True)
print("\nScores (sudden):\n", scores_sudden)

scores_gradual = np.load('results_gradual.npy', allow_pickle=True)
print("\nScores (gradual):\n", scores_gradual)

scores_incremental = np.load('results_incremental.npy', allow_pickle=True)
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



############## PRESENTING RESULTS ######################################################

def show_results(mean, std): 
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id])) 

print("\n\nResults (sudden):")
show_results(mean_sudden, std_sudden)
print("\nResults (gradual):")
show_results(mean_gradual, std_gradual)
print("\nResults (incremental):")
show_results(mean_incremental, std_incremental)



############# STATISTICAL ANALYSIS ######################################################

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))


def create_result_tables(scores):

    #creating t_statistic and p_value matrices
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print("\nt-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = ["UOB", "OOB", "OB", "SEA"]
    names_column = np.array([["UOB"], ["OOB"], ["OB"], ["SEA"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("\nt-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    #advantage matrics
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\n\nAdvantage:\n\n", advantage_table)

    #significance table
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\n\nStatistical significance (alpha = 0.05):\n\n", significance_table)

    #final results
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("\n\nStatistically significantly better:\n\n", stat_better_table)


#printing results
print("\n\n==============================================================================================")
print("Results for sudden drift:")
create_result_tables(scores_sudden)
print("\n\n==============================================================================================")
print("Results for gradual drift:")
create_result_tables(scores_gradual)
print("\n\n==============================================================================================")
print("Results for incremental drift:")
create_result_tables(scores_incremental)
