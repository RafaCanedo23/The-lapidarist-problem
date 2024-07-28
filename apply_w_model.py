import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

# Custom transformers
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Select columns and return DataFrame"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].copy()

class DataFrameWrapper(BaseEstimator, TransformerMixin):
    """Wrap the transformer to output a DataFrame with ordered columns"""
    def __init__(self, transformer, attribute_names):
        self.transformer = transformer
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        transformed_array = self.transformer.transform(X)
        return pd.DataFrame(transformed_array, columns=self.attribute_names)

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable, mapping in self.mappings.items():
            X[variable] = X[variable].map(mapping)
        return X

class RemoveOutliersSD(BaseEstimator, TransformerMixin):
    """Remove outliers based on standard deviation criteria"""
    def __init__(self, columns, n_std=3):
        self.columns = columns
        self.n_std = n_std

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            mean, std = X[column].mean(), X[column].std()
            cutoff = std * self.n_std
            lower, upper = mean - cutoff, mean + cutoff
            X = X[(X[column] >= lower) & (X[column] <= upper)]
        return X

# Define the mappings
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

# Define the pipeline
numeric_features = ['carat', 'depth', 'table', 'latitude', 'longitude ']
categorical_features = ['cut', 'color', 'clarity']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(mappings)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', MinMaxScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Correct order of columns for the model
correct_order = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'latitude', 'longitude ']

class ColumnReorder(BaseEstimator, TransformerMixin):
    """Reorder columns to the correct order"""
    def __init__(self, correct_order):
        self.correct_order = correct_order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.correct_order]

full_pipeline = Pipeline(steps=[
    ('remove_outliers', RemoveOutliersSD(columns=['depth', 'table', 'latitude', 'longitude '])),
    ('preprocessor', DataFrameWrapper(preprocessor, numeric_features + categorical_features)),
    ('reorder', ColumnReorder(correct_order))
])

# Apply transformer
fit_data = pd.read_csv('Data/cleaned_data.csv')
fit_data = fit_data.drop(['x', 'y', 'z'], axis=1)
full_pipeline.fit(fit_data)

# Transform the unseen data
unseen_data = pd.read_csv('Data/unseen_data.csv')
unseen_data = unseen_data.drop(['x', 'y', 'z'], axis=1)
transformed_data = full_pipeline.transform(unseen_data)

# Save the pipeline
joblib.dump(full_pipeline, 'Models/preprocessing_pipeline.pkl')

# Import predictive models
# Load the model and transformer
loaded_poly_model = joblib.load('Models/poly_model.pkl')
loaded_poly_trans = joblib.load('Models/poly_transformer.pkl')

# Transform new data and make predictions
X_new_poly = loaded_poly_trans.transform(transformed_data)
new_predictions = loaded_poly_model.predict(X_new_poly)

# Save unseen data + predictions
unseen_data['Price_pred'] = new_predictions
unseen_data.to_csv('Results/predictions_data.csv', index=False)