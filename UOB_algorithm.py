"""UOB improved algorithm"""
"""UOB -> Undersampling Online Bagging"""
"""UOB is an online bagging algorithm created in purpose of handle inbalanse classes """

#Python imported libraries
import numpy as np 
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

class UOB(BaseEnsemble, ClassifierMixin):
    """
    INPUT VALUES:
    a) current_example - current training example (x,y)
    b) positive_class_size - positive class size (w(+))
    c) negative_class_size - negative class size (w(-))
    d) predefined_time_decay_factor - param for data update support(rec. value = 0.9)
    """
    def __init__(self, ensemble_of_classifiers=None, predefined_time_decay_factor=0.9):
        """Initialization."""
        self.ensemble_of_classifiers=ensemble_of_classifiers
        self.predefined_time_decay_factor = predefined_time_decay_factor
        self.positive_class_size = 1
        self.negative_class_size = 1

    
    def update_class_size(self, y):
        if y==1:
            self.positive_class_size = self.positive_class_size*self.predefined_time_decay_factor + (1 - self.predefined_time_decay_factor)
        else:
            self.negative_class_size = self.negative_class_size*self.predefined_time_decay_factor + (1 - self.predefined_time_decay_factor)

    def partial_fit(self, X, y, classes=None):
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y

        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        #zipp x and y to current example
        #for each example update class size
        for current_example in zip(X,y):
            self.update_class_size(current_example[1])

            #improved UOB algorithm for each base learner m in M set
            for M in self.ensemble_of_classifiers:
                lambda_poisson_param = 0
                #if class is positive and positive class size is higher than negative class size
                if (current_example[1]==1) and (self.positive_class_size > self.negative_class_size):
                    lambda_poisson_param = self.negative_class_size / self.positive_class_size

                #if class is negative and negative class size is higher than positive class size
                if (current_example[1]==1) and (self.negative_class_size > self.positive_class_size):
                    lambda_poisson_param = self.positive_class_size / self.negative_class_size
                
                #if not previous conditions met set lambda parameter to one
                else:
                    lambda_poisson_param = 1
            
                #determine K value according to poisson distribution with lamda parameter
                K = np.random.poisson(lambda_poisson_param)

                #fit data
                if (K != 0):
                    M.partial_fit(self.X_, self.y_, self.classes_, sample_weight=K)
        return self

        #make prediction 
    def predict(self, X):
        support_matrix = np.array([M.predict_proba(X) for M in self.ensemble_of_classifiers])
        means = np.mean(support_matrix, axis=0)
        predict = np.argmax(means, axis=1)

        return predict

