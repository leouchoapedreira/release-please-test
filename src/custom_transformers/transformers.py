from collections import Counter

import pandas as pd

# from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

# NOTE: about the warning
# If you receive the following warning:
#
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
#
# The pandas package will send you to the following docs, that
# states that starting in pandas version 3.0, the copy-on-write will
# be default, so there's no need to worry about this except in extreme
# performance scenarios.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.copy_on_write = True


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for selecting columns in specified order
    """

    def __init__(self, columns: list[str], drop_cols: bool = False):
        self.columns = columns
        self.drop_cols = drop_cols

        column_counts = Counter(self.columns)
        duplicates = [col for col, count in column_counts.items() if count > 1]

        if duplicates:
            raise ValueError(
                f"Duplicate columns found in the columns list: {', '.join(duplicates)}."
            )

    def fit(self, X: pd.DataFrame, y=None):
        """
        fit
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        transform
        """
        if self.drop_cols:
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise KeyError(f"{missing_cols} not found in axis")

            return X.drop(columns=self.columns)
        else:
            return X[self.columns]
