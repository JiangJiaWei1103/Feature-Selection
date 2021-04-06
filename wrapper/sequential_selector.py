'''Wrapper Methods - Sequential Search in Feature Selection'''
# Authors: JiaWei Jiang

# Import packages
import os 
import json

import pandas as pd 
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

class SequentialSearch(SFS):
    '''Feature selector using wrapper method, selecting features through a sequential process.
    
    Parameters:
        estimator: sklearn estimators, the modelling algorithm (e.g. classifier, regressor)
        feature_range: int or tuple or str, number of features to be selected, explanation for different dtypes are as follows:
            int: a fixed number of features will be selected
            tuple: all feature numbers between the lower and upper bounds will be tried in sfs
            str: 
                "best": feature selector will return the feature subset with the best CV performance
                "parsimonious": the smallest feature subset within 1 standard error of CV performance will be selected
        method: str, the greedy methodology used in sequential feature selection, choices are as follows:
                "f": forward selection without floating
                "b": backward selection without floating
                "ff": forward selection with floating (conditional feature exclusion)
                "bf": backward selection with floating (conditional feature inclusion)
        scoring: str or callable (sklearn.metrics.Scorer), scoring metrics, the choices are as follows:
            1. See sklearn.metrics.SCORERS.keys() for scoring function choices with dtype str
            2. Callable object or function with signature scorer(estimator, X, y), returning the score
        cv_k: int, fold number for CV, specify {None, False, 0} to disable CV and sample number {n} to do LOOCV, default=5
            *Note: Stratified for classifier, and Regular for regressor
        n_jobs: int, number of CPUs to use for evaluating different feature subsets in parallel, default=-1 (use all CPUs)
        fixed_features: tuple, feature indexes or feature names indicating the features guaranteed to be present in selecting results, default=None
        
        #==========
        pre_dispatch
        clone_estimator
        cv isn't restricted to k-fold
        #==========
    
    Attributes:
        best_performance_: tuple, the best feature set containing feature names with its average CV score measured by scoring metrics
        step_report_: pd.DataFrame, a performance report in detail during the exhaustive feature selection 
    '''
    def __init__(self, estimator, feature_range, method, 
                 scoring, cv_k=5, n_jobs=-1):
        self.forward = True if method.startswith("f") else False
        self.floating = True if len(method) == 2 else False
        super().__init__(estimator=estimator,
                         k_features=feature_range,
                         forward=self.forward,
                         floating=self.floating,
                         scoring=scoring,
                         cv=cv_k,
                         n_jobs=n_jobs)
        #=================================
        # Set attributes to avoid AttributeError, raised from get_params(self, deep) in base.py
        '''Why???????????????????????????'''
        self.feature_range = feature_range
        self.cv_k = cv_k
        #=================================
        self.estimator_name = estimator.__class__.__name__
        self.scoring_fn = scoring if isinstance(scoring, str) else scoring.__name__
        
    def fit(self, X, y, groups=None):
        '''Run sequential feature selection using features from X with target y.
        
        Parameter:
            X: ndarray or pd.DataFrame, dataframe with raw features (columns) to be seleted, with shape (n_samples, n_features) 
            y: ndarray or pd.Series, the target values (e.g. class labels for classification, real number for regression), with shape (n_samples, )
            groups: array-like, group lables for the samples used while splitting the  dataset into train/test set, passed to the fit method of the cross-validator, with shape(n_samples, ) 
                * Directly set to None to disable the functionality.
                * Need this redundent parameter to deal with the error resulted from the function call of fit_transform which will call function fit with the parameter groups. Otherwise, the unexpected keyword argument 'groups' will be raised. 
        
        Return:
            self 
        '''
        self.feature_names = X.columns if hasattr(X, "iloc") else [_ for _ in range(X.shape[1])]
        super().fit(X=X, y=y, custom_feature_names=self.feature_names)
        
        # Assign attributes
        self.best_performance_ = (self.k_feature_names_, self.k_score_)
        self.step_report_ = pd.DataFrame.from_dict(self.get_metric_dict()).T
        self.step_report_.rename(columns={"cv_scores": f"cv_scores ({self.scoring_fn})", "avg_score": f"avg_score ({self.scoring_fn})"}, inplace=True)
        
        return self
        
    def transform(self, X):
        '''Return the best feature subset selected from X.
        
        To directly return the selecting results with type pd.DataFrame, this function is implemented by method overriding.
        
        Parameters:
            X: ndarray or pd.DataFrame, dataframe with raw features (columns) to be seleted, with shape (n_samples, n_features) 
        
        Return:
            X: pd.DataFrame, the best feature subset selected from X, with shape (n_samples, k_features), k is the number of selected features
        '''
        super()._check_fitted()
        X = X.values if hasattr(X, "iloc") else X
        X = X[:, self.k_feature_idx_]
        return pd.DataFrame(X, columns=self.k_feature_names_)
    
    def plot_performance(self, cv_band="std_dev"):
        '''Plot the performance for different number of features during selecting procedure.
        
        The blue line is the average CV performance measured by the pre-specified scoring function. The blue band indicates the distance of one standard deviation from the average CV performance.
        
        Parameters:
            err_interval: str, the kind of the error bar or confidence interval, choices are as follows:
                {"std_dev", "std_err", "ci", None}, default="std_dev"
        '''
        super()._check_fitted()   # Check if the selector has fitted to data
        plot_sfs(self.get_metric_dict(),
                 kind=cv_band)
        direction = "Forward" if self.forward else "Backward"
        floating = "Floating" if self.floating else ""
        plt.title(f"{direction}-{floating} Selection with Feature Range {self.feature_range}")
        plt.ylabel(f"{self.scoring_fn} +/- {cv_band}")
        plt.show()
    
    def __repr__(self):
        sfs_params = {
            "estimator": self.estimator_name,
            "scoring": self.scoring_fn,
            "k_feature": self.feature_range,
        }
        sfs_params = json.dumps(sfs_params, indent=2)
        return f"Sequentail feature selection:\n{sfs_params}"