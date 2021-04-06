'''Simple Filter Methods in Feature Selection'''
# Authors: JiaWei Jiang

# Import packages
import os 
import math 
from abc import abstractmethod
from warnings import warn 

import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import as_float_array
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import KFold

__all__ = [
    'DuplicateFilter',
    'QuasiConstFilter',
    'HighCorrFilter',
    'KBestFilter',
]

class _BaseFilter(SelectorMixin, BaseEstimator):
    '''Base class for feature selector using filter methods.
    '''
    
    @abstractmethod
    def _get_support_mask(self):
        '''Get the boolean mask indicating which features are selected.
        
        Return: ndarray or pd.Series, a mask indicating the features are selected or not, with shape (n_features, )
        '''
    
    def transform(self, X):
        '''Reduce X to the selected features.
        
        Parameters:
            X: ndarray or pd.DataFrame, input samples with shape (n_samples, n_features)
        Return:
            X_selected: ndarray or pd.DataFrame, input samples containing ony the selected features with shape (n_samples, n_features)
        '''
        print(f"=====Start filtering features!=====")
        if not hasattr(X, "iloc"): 
            # If X isn't a pd.DataFrame
            X = pd.DataFrame(X) 
        mask = self.get_support()
        if not mask.any():
            # No features selected (all filtered out)
            warn("No features were selected. All of the features have been filtered out.", UserWarning)
            return pd.DataFrame()
        if len(mask) != X.shape[1]:
            # If X has been modified during fitting.
            raise ValueWarninig("X has a different number of features during fitting.")
        features_selected = [feature for feature, selected in zip(X.columns, mask) if selected]
        print("=====Filtering done!=====")
        return X.loc[:, features_selected] 
          
class DuplicateFilter(_BaseFilter):
    '''Feature selector using filter method, removing all the duplicated features with only one left.
    
    This feature selection algorithm is used as an "multivariate" one, which takes the interaction between the features into consideration.
    
    Parameters:
        keep: str, feature that will be kept based on the order of the occurrences, choices are as follows:
            {"first", "last"}, default="first"
    
    Attributes:
        duplicates_: list, groups of the duplicated features     
    '''
    def __init__(self, keep="first"):
        self.keep = keep
        
    def fit(self, X, y=None):
        '''Find and mark the duplicated features from X.
        
        Parameters:
            X: ndarray or pd.DataFrame, dataframe with raw features (columns) to be determined whether the duplicates exist or not, with shape (n_samples, n_features)
            
        Return:
            self
        '''
        print(f"=====Start finding all the duplicated features!=====")
        if not hasattr(X, "iloc"): 
            # if X isn't a pd.DataFrame
            X = pd.DataFrame(X) 
        self.feature_names = X.columns
        X_T = X.T   # Get the transpose of the dataframe X
        duplicated = X_T.duplicated(keep=False)   # Mark duplicated features with boolean values
        features_duplicated = [duplicated.index[i] for i in range(len(duplicated)) if duplicated[i]]
        df_duplicated = X_T.loc[features_duplicated, :]   # Datafarme containing all the duplicated features
        groups_duplicated = df_duplicated.groupby(by=list(df_duplicated), axis=0)   # Group the duplicated features by all the samples
        self.duplicates_ = []
        for i, (n, g) in enumerate(groups_duplicated):
            duplicated_feature_group = list(g.index)
            print(f"Duplicated feature group{i}: {duplicated_feature_group}")
            self.duplicates_.append(duplicated_feature_group)
        
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self)   
        
        uniques = pd.Series(np.ones(len(self.feature_names), dtype=bool), index=self.feature_names)   # Mask for determining the uniqueness of features
        for duplicate_feature_group in self.duplicates_:
            redundents = duplicate_feature_group[:-1] if self.keep == "last" else duplicate_feature_group[1:]
            for redundent in redundents:
                uniques[redundent] = False
        return uniques
    
class QuasiConstFilter(_BaseFilter):
    '''Feature selector using filter method, removing all the constant or quasi-constant features, using the methods of variance testing or equivalence testing.
    
    This feature selection algorithm is used as an "univariate" one, which considers only a single feature at once and can be used for unsupervised learning.
    
    Parameters:
        method: str, method used to determine the constant or quasi-constant features, the choices are as follows:
            {"var", "proportion"}, default="var"
        thres: float, variance threshold or value proportion threshold used to determine whether to filter out the feature, default=0.0
        
    Attributes:
        variances_: pd.Series, the variance of each feature, with shape (n_features, ). This is used when you specify the method as "var".
        equivalences_: pd.Series, the highest proportion for all the feature values of each features, with shape (n_features, ). This is used when you specify the method as "proportion".
    '''
    def __init__(self, method="var", thres=0.0):
        self.method = method
        self.thres = thres
    
    def fit(self, X, y=None):
        '''Calculate the determining criteria, based on variances or proportions of feature values from X.  

        Parameters:
            X: ndarray or pd.DataFrame, dataframe with raw features (columns) from which to calculate variances or proportions of feature values, with shape (n_samples, n_features)
            
        Return:
            self
        '''
        print(f"=====Start calculating {self.method} for determining constant or quasi-constant features!=====")
        self.feature_names = X.columns if hasattr(X, "iloc") else [_ for _ in range(len(X))]
        sample_num = len(X)
        if self.method == "var":
            selector = VarianceThreshold(threshold=self.thres)
            selector.fit(X)
            self.variances_ = pd.Series(selector.variances_, index=self.feature_names)
        elif self.method == "proportion":
            self.equivalences_ = []
            X = np.array(X) if hasattr(X, "iloc") else X   # Convert the dataframe to ndarray
            for col_idx in range(X.shape[1]):
                max_proportion = float(np.unique(X[:, col_idx], return_counts=True)[1].max()) / sample_num
                self.equivalences_.append(max_proportion)
            self.equivalences_ = pd.Series(self.equivalences_, index=self.feature_names)
        
        return self
            
    def _get_support_mask(self):
        check_is_fitted(self)
        
        if self.method == "var":
            return self.variances_ > self.thres
        elif self.method == "proportion":
            return self.equivalences_ < self.thres
        
class HighCorrFilter(_BaseFilter):
    '''Feature selector using filter method, if two or more features are highly correlated, remain only one of the features and filter out the others.
    
    This feature selection algorithm is used as an "multivariate" one, which takes the interaction between the features into consideration.
    
    Parameters:
        method: str, method to calculate the correlations, choices are as follows:
            {"pearson", "kendall", "spearman"}, default="pearson"
        cor_thres: float, correlation threshold to determine whether to filter out the features, default=0.95
        
    Attributes:
        correlations_: pd.DataFrame, the 2-D correlation matrix 
    '''
    def __init__(self, method="pearson", cor_thres=0.95):
        self.method = method
        self.cor_thres = cor_thres
    
    def fit(self, X, y=None):
        '''Calculate the correlations between features from X.
        
        Parameters:
            X: ndarray or pd.DataFrame, dataframe with raw features (columns) from which to calculate variances or proportions of feature values, with shape (n_samples, n_features)
            
        Return:
            self
        '''
        print(f"=====Start calculating correlations between features using {self.method} as measuring method!=====")
        if not hasattr(X, "iloc"): 
            # if X isn't a pd.DataFrame
            X = pd.DataFrame(X)
        self.feature_names = X.columns
        self.correlations_ = X.corr(method=self.method)

        return self 
    
    def _get_support_mask(self):
        check_is_fitted(self)
        
        corr = self.correlations_.abs()
        corr_U = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))   # Upper triangular matrix of correlations
        weak_corrs = corr_U.apply(lambda x: not any(x > self.cor_thres), axis=0)   # Mask indicating whether the feature has high correlation with others, True if the linear relationship is weak
        return weak_corrs  
    
class KBestFilter(_BaseFilter):
    '''Feature selector using filter method, a statistical test is implemented in order to select the top k best features based on the score measured by scoring function.
    
    This feature selection algorithm is used as an "???" one, which considers the interaction between each single feature from X and target y.
    
    Parameters:
        score_fn: callable, function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores, choices are as follows:
            {f_regression, mutual_info_regression, f_classif, mutual_info_classif, chi2}, default=f_regression
        k: int or "all", the number of features to be filtered, default=10
        
    Attributes:
        score_report_: pd.DataFrame, a report containing the scoring results of the specified score function, with shape (n_features, 2) if there's pvalues_ returned by score_fn; otherwise, only (n_features, 1)
    '''
    def __init__(self, score_fn=f_regression, k=10): 
        self.score_fn = score_fn
        self.k = k
    
    def fit(self, X, y):
        '''Run score function on (X, y) and get the desired attributes (e.g. scores_, pvalues_).
        
        Parameters:
            X: ndarray or pd.DataFrame, dataframe with raw features (columns) from which to run score function with target y, with shape (n_samples, n_features)
            y: ndarray or pd.Series, the target values (e.g. class labels for classification, real number for regression), with shape (n_samples, )
            
        Return:
            self
        '''
        print(f"=====Start running score function {self.score_fn.__name__} on feature X and target y!=====")
        self.feature_names = X.columns if hasattr(X, "iloc") else [_ for _ in range(len(X))]
        selector = SelectKBest(self.score_fn, self.k)
        selector.fit(X, y)
        
        # Assign attributes
        self.score_report_= pd.DataFrame(index=self.feature_names)
        self.score_report_["score"] = selector.scores_
        if selector.pvalues_ is not None:
            self.score_report_["p value"] = selector.pvalues_
        
        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        if self.k == 'all':
            return np.ones(len(self.score_report_), dtype=bool)
        elif self.k == 0:
            return np.zeros(len(self.score_report_), dtype=bool)
        else:
            scores = as_float_array(self.score_report_["score"], copy=True, force_all_finite='allow-nan')   # Convert the array -like to an array of floats.
            scores[np.isnan(scores)] = np.finfo(scores.dtype).min
            high_scores = np.zeros(scores.shape, dtype=bool)
            high_scores[np.argsort(scores, kind="mergesort")[-self.k:]] = 1
            return high_scores