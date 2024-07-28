import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# functions library
def remove_outliers_sd(df, column, n_std=3):
    """
    Filtering & cleaning outliers
    """
    mean, std = df[column].mean(), df[column].std()
    cutoff = std * n_std
    lower, upper = mean - cutoff, mean + cutoff
    return df[(df[column] >= lower) & (df[column] <= upper)]

def mean_imputing(df, column):
    """
    Replacing missing values in a vector with the mean of the vector.
    Use it when the vector follows a normal distribution.
    """
    mean = df[column].mean()
    df[column] = df[column].fillna(mean)

def meadian_imputing(df, column):
    """
    Replacing missing values in a vector with the median of the vector.
    Use it when the vector has skewness.
    """
    median = df[column].median()
    df[column] = df[column].fillna(median)

def ordinal_encoding(df, variable, mapping):
    """ 
    Generate ordinal encoding for categorical values

    Parameters:
    df (pd.DataFrame): Input DataFrame
    variable (str): name of categorical variable
    map (dict): encoding map
    """
    df[variable] = df[variable].map(mapping)

# Creating the ordinal encoder maps
cut_mapping = {
    'Fair': 1,
    'Good': 2,
    'VeryGood': 3,
    'Premium': 4,
    'Ideal': 5
}

color_mapping = {
    'J': 1,
    'I': 2,
    'H': 3,
    'G': 4,
    'F': 5,
    'E': 6,
    'D': 7
}

clarity_mapping = {
    'I1': 1,
    'SI2': 2,
    'SI1': 3,
    'VS2': 4,
    'VS1': 5,
    'VVS2': 6,
    'VVS1': 7,
    'IF': 8
}

mappings = {
    'cut': cut_mapping,
    'color': color_mapping,
    'clarity': clarity_mapping
}

# Unseen data
