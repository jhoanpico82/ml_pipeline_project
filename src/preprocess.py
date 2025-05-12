# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path):
    """
    Carga los datos y elimina las filas con valores nulos en 'Global_Sales' y 'Year'.
    """
    df = pd.read_csv(path)
    df.dropna(subset=['Global_Sales'], inplace=True)
    df = df[df['Year'].notna()]
    return df


def split_features_target(df):
    """
    Separa las características (X) y la variable objetivo (y).
    """
    X = df[
        ['Platform', 'Year', 'Genre', 'Publisher',
         'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    ]
    y = df['Global_Sales']
    return X, y


def get_preprocessor():
    """
    Devuelve un preprocesador con transformaciones para las características
    numéricas y categóricas.
    """
    categorical_features = ['Platform', 'Genre', 'Publisher']
    numerical_features = [
        'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'
    ]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

    return preprocessor



