from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import numpy as np


def get_median_price_pipe(df):
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    # print(list(num_cols))
    # print(list(cat_cols))

    num_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "scaler",
                MinMaxScaler(),
            ),  # Nomralise feature importance (for distance based models)
        ]
    )
    cat_trans = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # print(num_trans)
    # print(cat_trans)

    def log_transform(x):
        return np.log1p(x - x.min() + 1)

    log_custom = FunctionTransformer(
        log_transform, feature_names_out="one-to-one"
    )
    log_trans_cols = [
        # "elevation",
        # "distanceToWatercourse",
        # "sectorAnimalToHumanRatio",
    ]
    log_trans = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # median should work better for skewed data
            ("log_transform", log_custom),
        ]
    )

    prep_pipe = ColumnTransformer(
        transformers=[
            ("log", log_trans, log_trans_cols),
            ("num", num_trans, num_cols.difference(log_trans_cols)),
            ("cat", cat_trans, cat_cols),
        ]
    )

    return prep_pipe


def get_risk_label_pipe(df):
    """
    Define a unified preprocessing pipeline for all data.
    """

    # num_cols = df.select_dtypes(include=np.number).columns
    # cat_cols = df.select_dtypes(include="object").columns
    # Define numeric log transformation
    def log_transform(x):
        # Apply log transformation to ensure all values are positive
        return np.log1p(x - np.min(x, axis=0) + 1)

    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="mean"),
            ),  # Handle missing values
            (
                "log_transform",
                FunctionTransformer(log_transform, validate=True),
            ),  # Log transform
            ("scaler", StandardScaler()),  # Standardize
        ]
    )

    # Define preprocessing for categorical features
    categorical_transformer = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )

    # Combine preprocessing for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["elevation"]),  # Numerical columns
            (
                "cat",
                categorical_transformer,
                ["soilType"],
            ),  # Categorical columns
        ],
        remainder="passthrough",  # Pass through other columns unchanged
    )
    return preprocessor


def get_local_authority_pipeline():
    numerical_features = ["easting", "northing"]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
        ]
    )

    return preprocessor
