from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DeleteNAN(BaseEstimator, TransformerMixin):
    def __init__(self, nan_char):
        self.nan_char = nan_char
    
    def set_nan_char(self, char):
        self.nan_char = char

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        data_nan= X.replace(self.nan_char,np.nan)
        data_nan.dropna(how="any", inplace=True)
        return data_nan


class ExchangeNan(BaseEstimator, TransformerMixin):
    def __init__(self, nan_char, ex_char):
        self.nan_char = nan_char
        self.ex_char = ex_char

    def set_nan_char(self, char):
        self.nan_char = char

    def set_ex_char(self, char):
        self.ex_char = char
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        data_nan= X.replace(self.nan_char,self.ex_char)
        return data_nan




class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, value):
        self.value = value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.att = list(X.select_dtypes(self.value).columns)
        return X[self.att]