"""Undersampling-based Online Bagging."""

"""© Copyright 2019, P. Ksieniewicz, P. Zyblewski"""
"""KSSK / Wrocław University of Science and Technology"""
"""https://w4k2.github.io/stream-learn"""


import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

class UOB(BaseEnsemble, ClassifierMixin):
    """
    Undersampling-Based Online Bagging.
    """
    #funkcja init 
    #time_decay_factor = 0.9
    #base_estmator ???
    #n_estimators = 5 ???
    #input: zbiory Yminority, Ymajority z M uczacymi
    def __init__(self, base_estimator=None, n_estimators=5, time_decay_factor=0.9):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.time_decay_factor = time_decay_factor
    #funkcja odpowiedzialna za przygotowanie i dopasowanie danych
    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self
    #dopasowanie oraz klonowanie z uzwglednieniem odpowiedniego estymatora
    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        # sprawdzanie czy klasa instnieje oraz zwracanie unikalnych etykiet w inwersji
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # time decayed class sizes tracking
        if not hasattr(self, "last_instance_sizes"):
            self.current_tdcs_ = np.zeros((1, 2))    #gdy nie ma tdcs uzupelnianie zerami 
        else:
            self.current_ctdcs_ = self.last_instance_sizes #w innym przypadku przypisanie do obecnego tdcs ostatniego tdcs

        self.chunk_tdcs = np.ones((self.X_.shape[0], self.classes_.shape[0]))
        #okreslanie rozmiaru klasy dla zboriu etykiet 
        for iteration, label in enumerate(self.y_):
            if label == 0: #????
                self.current_tdcs_[0, 0] = (
                    self.current_tdcs_[0, 0] * self.time_decay_factor
                ) + (1 - self.time_decay_factor)  #etykieta xt jest ck
                self.current_tdcs_[0, 1] = (
                    self.current_tdcs_[0, 1] * self.time_decay_factor
                ) #etykieta xt nie jest ck ???
            else:
                self.current_tdcs_[0, 1] = (
                    self.current_tdcs_[0, 1] * self.time_decay_factor
                ) + (1 - self.time_decay_factor) #etykieta xt jest ck
                self.current_tdcs_[0, 0] = (
                    self.current_tdcs_[0, 0] * self.time_decay_factor
                ) #etykita xt nie jest ck

            self.chunk_tdcs[iteration] = self.current_tdcs_ #dla kazdej iteracji M klas wejsciowych przypisanie obecnego tdcs 

        self.last_instance_sizes = self.current_tdcs_ #aktualizacja rozmiaru klasy w kazdej iteracji

        # improved UOB
        # assumptions: Y(-) -> klasa negatywna, Y(+) -> klasa pozytywna 
        # w+/ chuck_tdcs [instance][1] , w-/ chuck_tdcs [instance][0]
        self.weights = [] #tablica wag
        for instance, label in enumerate(self.y_):
            if (
                label == 1 #klasa pozytywna???
                and self.chunk_tdcs[instance][1] > self.chunk_tdcs[instance][0] # w(+)/w(-)
            ):
                lmbda = self.chunk_tdcs[instance][0] / \
                    self.chunk_tdcs[instance][1]        # obliczanie wartosci lambda do poissona 
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)] #wyznaczanie wartosci k 
                )
            elif (
                label == 0 #klasa negatywna 
                and self.chunk_tdcs[instance][0] > self.chunk_tdcs[instance][1]
            ):
                lmbda = self.chunk_tdcs[instance][1] / \
                    self.chunk_tdcs[instance][0]
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )
            else: #jezeli lambda rowna jeden
                lmbda = 1
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )
            self.weights.append(K) #rozszerzanie tablicy wag

        self.weights = np.asarray(self.weights).T #formatowanie tablicy 

        for w, base_model in enumerate(self.ensemble_):
            base_model.partial_fit(
                self.X_, self.y_, self.classes_, sample_weight=self.weights[w]
            )

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X) #macierz wsparcia 
        average_support = np.mean(esm, axis=0) #wsparcie wartosci sredniej
        prediction = np.argmax(average_support, axis=1) #wyznaczenie wartosci maksymalnej predykcji 

        # Return prediction
        return self.classes_[prediction] #zwrocenie wyznaczonej predykcji 
