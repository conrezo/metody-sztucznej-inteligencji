import numpy as np
import strlearn as sl
from strlearn.ensembles import OOB, UOB, SEA, OnlineBagging
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_ind
from tabulate import tabulate
from ourUOB import ourUOB


############## CLASSIFICATORS AND METRICS #############################################

#classificators
clfs = {   
    'UOB': UOB(base_estimator=GaussianNB(), n_estimators=5),
    'ourUOB': ourUOB(base_estimator=GaussianNB(), n_estimators=5),
    'OOB': OOB(base_estimator=GaussianNB(), n_estimators=5),
    'OB': OnlineBagging(base_estimator=GaussianNB(), n_estimators=5),
    'SEA': SEA(base_estimator=GaussianNB(), n_estimators=5)
}


#metrics names
metrics_names = ["G-mean"] 
#other:  F1 score

#metrics
metrics = [sl.metrics.geometric_mean_score_1] 
#other: sl.metrics.f1_score



########### GENERATING STREAMS ########################################################

n_chunks_value = 200

def generating_streams(random_state_value):
    #generate sudden drift
    stream_sudden = sl.streams.StreamGenerator(n_chunks=n_chunks_value,
                                               chunk_size=100,
                                               n_classes=2,
                                               n_drifts=1,
                                               n_features=10,
                                               weights=[0.2, 0.8],
                                               random_state=random_state_value)

    #generatre gradual drift
    stream_gradual = sl.streams.StreamGenerator(n_chunks=n_chunks_value,
                                                chunk_size=100,
                                                n_classes=2,
                                                n_drifts=1,
                                                n_features=10,
                                                weights=[0.2, 0.8],
                                                random_state=random_state_value,
                                                concept_sigmoid_spacing=5)

    #generate incremental drift
    stream_incremental = sl.streams.StreamGenerator(n_chunks=n_chunks_value,
                                                    chunk_size=100,
                                                    n_classes=2,
                                                    n_drifts=1,
                                                    n_features=10,
                                                    weights=[0.2, 0.8],
                                                    random_state=random_state_value,
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

    return evaluator_sudden.scores, evaluator_gradual.scores,  evaluator_incremental.scores


random_sate_list = [10, 1410, 21, 653, 1234, 190, 859, 329, 2137, 929]

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


#saving results (evaluator scores) to file
def save_to_file(full_scores, drift_name: str):
    np.save('./results/results_' + drift_name, full_scores)

save_to_file(full_scores_sudden, "sudden")
save_to_file(full_scores_gradual, "gradual")
save_to_file(full_scores_incremental, "incremental")




################ DATA ANALYSIS ######################################################

#reading_result_from_file 
scores_sudden = np.load('./results/results_sudden.npy', allow_pickle=True)
print("\n\nScores (sudden):\n", scores_sudden)

scores_gradual = np.load('./results/results_gradual.npy', allow_pickle=True)
print("\n\nScores (gradual):\n", scores_gradual)

scores_incremental = np.load('./results/results_incremental.npy', allow_pickle=True)
print("\n\nScores (incremental):\n", scores_incremental)


#mean scores
def mean_calculate(mean):
    means_list = []
    for j in range(0, len(clfs)):
        sudden_clf_mean = []
        for i in range(j, len(mean) - 1, len(clfs)):
            arr = np.append(sudden_clf_mean, mean[i], axis=0)
            sudden_clf_mean = arr
        means = np.mean(sudden_clf_mean)
        means_list.append(means)
    return means_list

#showing mean
mean_sudden_array = np.mean(scores_sudden, axis=1)
mean_sudden = mean_calculate(mean_sudden_array)
print("\n\nMean (sudden):\n", mean_sudden)

mean_gradual_array = np.mean(scores_gradual, axis=1)
mean_gradual = mean_calculate(mean_gradual_array)
print("\nMean (gradual):\n", mean_gradual)

mean_incremental_array = np.mean(scores_incremental, axis=1)
mean_incremental = mean_calculate(mean_gradual_array)
print("\nMean (incremental):\n", mean_incremental)


#std scores
def std_calculate(std):
    std_list = []
    for j in range(0, len(clfs)):
        std_clf = []
        for i in range(j, len(std) - 1, len(clfs)):
            arr = np.append(std_clf, std[i], axis=0)
            std_clf = arr
        stds = np.std(std_clf)
        std_list.append(stds)
    return std_list

#showing std
std_sudden_array = np.std(scores_sudden, axis=1)
std_sudden = std_calculate(std_sudden_array)
print("\n\nStd (sudden):\n", std_sudden)

std_gradual_array = np.std(scores_gradual, axis=1)
std_gradual = std_calculate(std_gradual_array)
print("\nStd (gradual):\n", std_gradual)

std_incremental_array = np.std(scores_incremental, axis=1)
std_incremental = std_calculate(std_incremental_array)
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

#defining alfa value
alfa = .05

#setting up t_statistic and p_value array filled up with zeros
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))


#creating results tabels
def create_result_tables(scores):

    #creating t_statistic and p_value matrices
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print("\nt-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = ["UOB", "ourUOB", "OOB", "OB", "SEA"]
    names_column = np.array([["UOB"], ["ourUOB"], ["OOB"], ["OB"], ["SEA"]])
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



#presenting final results
print("\n\n==============================================================================================")
print("Results for sudden drift:")
create_result_tables(scores_sudden)
print("\n\n==============================================================================================")
print("Results for gradual drift:")
create_result_tables(scores_gradual)
print("\n\n==============================================================================================")
print("Results for incremental drift:")
create_result_tables(scores_incremental)
