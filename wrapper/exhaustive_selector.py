'''Wrapper Methods - Exhaustive Search in Feature Selection'''
# Authors: JiaWei Jiang

# Import packages
import os 
import json

import pandas as pd 
import matplotlib.pyplot as plt
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
 
class ExhaustiveSearch(EFS):
    '''Feature selector using wrapper method, selecting features through a brute-force process.
    
    Parameters:
        estimator: sklearn estimators, the modelling algorithm (e.g. classifier, regressor)
        feature_range: tuple, two numbers specifying the lower and upper bounds of the number of featurs users want to select
        scoring: str or sklearn.metrics.Scorer, scoring metrics, choices are as follows:
            1. See sklearn.metrics.SCORERS.keys() for scoring function choices with dtype str
            2. Callable object or function with signature scorer(estimator, X, y), returning the score
        cv_k: int, fold number for CV, specify {None, False, 0} to disable CV and sample number {n} to do LOOCV, default=5
            *Note: Stratified for classifier, and Regular for regressor
        n_jobs: int, number of CPUs to use for evaluating different feature subsets in parallel, default=-1 (use all CPUs)
        
        #==========
        pre_dispatch
        clone_estimator
        cv isn't restricted to k-fold
        #==========
    
    Attributes:
        best_performance_: tuple, the best feature set containing feature names with its average CV score measured by scoring metrics
        step_report_: pd.DataFrame, a performance report in detail during the exhaustive feature selection 
    '''
    def __init__(self, estimator, feature_range, scoring, 
                 cv_k=5, n_jobs=-1):
        super().__init__(estimator=estimator,
                         min_features=feature_range[0],
                         max_features=feature_range[1],
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
        '''Run exhaustive feature selection using features from X with target y.
        
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
        self.best_performance_ = (self.best_feature_names_, self.best_score_)
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
        X = X[:, self.best_idx_]
        return pd.DataFrame(X, columns=self.best_feature_names_)
    
    def plot_performance(self):
        '''Plot the performance for each combination of features.
        
        The blue line is the average CV performance measured by the pre-specified scoring function. The blue band indicates the distance of one standard deviation from the average CV performance.
        '''
        super()._check_fitted()   # Check if the selector has fitted to data
        performance_dict = self.get_metric_dict()

        fig = plt.figure()
        comb_sorted = sorted(performance_dict.keys())   # Sort the performance dictionary by the index of feature combination
        avg_score = [performance_dict[comb]['avg_score'] for comb in comb_sorted]

        upper, lower = [], []   # Upper and lower bands evaluated by the standard deviation
        for comb in comb_sorted:
            upper.append(performance_dict[comb]['avg_score'] +
                         performance_dict[comb]['std_dev'])
            lower.append(performance_dict[comb]['avg_score'] -
                         performance_dict[comb]['std_dev'])

        plt.fill_between(comb_sorted,
                         upper,
                         lower,
                         alpha=0.2,
                         color='blue',
                         lw=1)

        plt.plot(comb_sorted, avg_score, color='blue', marker='o')
        plt.ylabel(f'{self.scoring_fn} +/- Standard Deviation')
        plt.xlabel('Feature Combination')
        plt.xticks(comb_sorted, 
                   [str(performance_dict[comb]['feature_names']) for comb in comb_sorted], 
                   rotation=90)
        plt.show()
        
    def __repr__(self):
        efs_params = {
            "estimator": self.estimator_name,
            "scoring": self.scoring_fn,
            "min_features": self.feature_range[0],
            "max_features": self.feature_range[1]
        }
        efs_params = json.dumps(efs_params, indent=2)
        return f"Exhaustive feature selection:\n{efs_params}" 
    
    '''
        * Implementaion of EFS with GridSearch
    '''