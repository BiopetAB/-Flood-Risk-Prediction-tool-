"""Example module in template package."""

# flake8: noqa: E501

import os
from collections.abc import Sequence
from typing import List, Optional

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .prep_pipes import (
    get_median_price_pipe,
    get_risk_label_pipe,
    get_local_authority_pipeline,
)


from .geo import *  # noqa: F401, F403

__all__ = [
    "Tool",
    "_data_dir",
    "_example_dir",
    "flood_class_from_postcode_methods",
    "flood_class_from_location_methods",
    "house_price_methods",
    "local_authority_methods",
    "historic_flooding_methods",
]

_data_dir = os.path.join(os.path.dirname(__file__), "resources")
_example_dir = os.path.join(os.path.dirname(__file__), "example_data")


# dictionaries with keys of short name and values of long name of
# classification/regression methods

# You should add your own methods here
flood_class_from_postcode_methods = {
    "all_zero_risk": "All zero risk",
    "rf_cls_risk_postcode": "Flood Risk from Postcode - Random Forest Classifier",
    "knn_cls_risk_postcode": "Flood Risk from Postcode - KNeighbors Classifier",
}
flood_class_from_location_methods = {
    "all_zero_risk": "All zero risk",
    "rf_cls_risk_location": "Flood Risk from Location - Random Forest Classifier",
    "knn_cls_risk_location": "Flood Risk from Location - KNeighbors Classifier",
}
historic_flooding_methods = {
    "all_false": "All False",
    "rf_cls_historic": "Historically Flooded - Random Forest Classifier",
}
house_price_methods = {
    "all_england_median": "All England median",
    "rf_reg_house_price": "House Price - Random Forest Regressor",
}
local_authority_methods = {
    # "all_nan": "All NaN",
    # "knn_cls_local_authority": "KNeighbors",
    "rf_cls_local_authority": "Local Authority - Random Forest Classifier",
}

IMPUTATION_CONSTANTS = {
    "soilType": "Unsurveyed/Urban",
    "elevation": 60.0,
    "nearestWatercourse": "",
    "distanceToWatercourse": 80,
    "localAuthority": np.nan,
}
flood_event_risk_table = {
    1: 0.001,
    2: 0.02,
    3: 0.005,
    4: 0.01,
    5: 0.02,
    6: 0.03,
    7: 0.05,
}


class Tool(object):
    def __init__(
        self,
        labelled_unit_data: str = "",
        unlabelled_unit_data: str = "",
        sector_data: str = "",
        district_data: str = "",
        station_data: str = "",
        additional_data: dict = {},
    ):
        """
        Parameters
        ----------
        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes. ()

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes. (postcodes)

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additional .csv files containing addtional
            information on households.

        postcode_data : str, optional
            Filename of a .csv file containing geographic location data for
            postcodes.

        station_data : str, optional
            Filename of a .csv file containing geographic location data for
            stations.
        """
        # Set defaults if no inputs provided
        if labelled_unit_data == "":
            labelled_unit_data = os.path.join(
                _data_dir, "postcodes_labelled.csv"
            )
        if unlabelled_unit_data == "":
            unlabelled_unit_data = os.path.join(
                _example_dir, "postcodes_unlabelled.csv"
            )
        if sector_data == "":
            sector_data = os.path.join(_data_dir, "sector_data.csv")
        if district_data == "":
            district_data = os.path.join(_data_dir, "district_data.csv")
        if station_data == "":
            station_data = os.path.join(_data_dir, "stations.csv")

        self._postcodedb = pd.read_csv(labelled_unit_data)
        self._districtdb = pd.read_csv(district_data)
        self._sectordb = pd.read_csv(sector_data)
        self._stationdb = pd.read_csv(station_data)
        self._unlabelled_unit_datadb = pd.read_csv(unlabelled_unit_data)
        self._combined_datadb = self.combine_resource_data(
            self._districtdb.copy(),
            self._postcodedb.copy(),
            self._sectordb.copy(),
            self._stationdb.copy(),
        )  # this method is stupid as it modifies the arguments.

        # Concat labelled and unlabelled data
        df = pd.concat([self._postcodedb, self._unlabelled_unit_datadb])
        df = df.drop_duplicates(subset=["postcode"])
        df = df.drop(
            columns=["medianPrice", "riskLabel", "historicallyFlooded"]
        )
        df["postcodeDistrict"] = df["postcode"].str.split(" ").str[0]
        df["postcodeArea"] = df["postcodeDistrict"].str.extract(
            r"([a-zA-Z]+)", expand=False
        )
        df["postcodeSector"] = (
            df["postcode"].str.split(" ").str[0]
            + " "
            + df["postcode"].str.split(" ").str[1].str[0]
        )
        self._lookupdb = df

        self._processed_unlabelled_unit_datadb = self.extract_postcode_info(
            self._unlabelled_unit_datadb
        )

        dt_params = {
            "max_depth": [3, 5, 7, 9],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }
        rf_params = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 5, 7, 9],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }
        knn_params = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
        }

        ln_reg = {"estimator": LinearRegression(), "param_grid": {}}
        rf_cls = {
            "estimator": RandomForestClassifier(),
            "param_grid": rf_params,
        }
        rf_reg = {
            "estimator": RandomForestRegressor(),
            "param_grid": rf_params,
        }
        knn_cls = {
            "estimator": KNeighborsClassifier(),
            "param_grid": knn_params,
        }

        self._models = {
            "rf_cls_risk_postcode": {
                **rf_cls,
                "prep_pipe_factory": get_risk_label_pipe,
                "prep_pipe": None,
            },
            "knn_cls_risk_postcode": {
                **knn_cls,
                "prep_pipe_factory": get_risk_label_pipe,
                "prep_pipe": None,
            },
            "rf_cls_risk_location": {
                **rf_cls,
                "prep_pipe_factory": get_risk_label_pipe,
                "prep_pipe": None,
            },
            "knn_cls_risk_location": {
                **knn_cls,
                "prep_pipe_factory": get_risk_label_pipe,
                "prep_pipe": None,
            },
            "rf_cls_historic": {
                **rf_cls,
                "prep_pipe_factory": get_median_price_pipe,
                "prep_pipe": None,
            },
            "rf_reg_house_price": {
                "estimator": Pipeline(
                    [
                        (
                            "model",
                            TransformedTargetRegressor(
                                regressor=RandomForestRegressor(),
                                func=np.log1p,
                                inverse_func=np.expm1,
                            ),
                        )
                    ]
                ),
                # model__regressor__n_estimators
                "param_grid": {
                    "model__regressor__n_estimators": [50, 100, 150, 200],
                    "model__regressor__max_depth": [3, 5, 7, 9],
                    "model__regressor__min_samples_split": [2, 4, 6, 8, 10],
                    "model__regressor__min_samples_leaf": [1, 2, 3, 4, 5],
                },
                "prep_pipe_factory": get_median_price_pipe,
                "prep_pipe": None,
            },
            "knn_cls_local_authority": {
                **knn_cls,
                "prep_pipe": get_local_authority_pipeline,
            },
            "rf_cls_local_authority": {
                **rf_cls,
                "prep_pipe": get_local_authority_pipeline,
            },
        }

    def fit(
        self,
        models: list = [],
        update_labels: str = "",
        update_hyperparameters: bool = False,
        **kwargs,
    ):
        """
        Fit/train models using a labelled set of samples for all sets of combinations of the three tasks riskLabel, medianPrice, historicallyFlooded.

        # NOTE - the reliability of this function depends on the 'update_labels' data to be in the same format as the sample data provided in 'postcodes_labelled.csv'
        # this directly affects the assignment of the 'data' variable

        Parameters
        ----------
        models : list
            List of model keys to train.
        update_labels : str
            Path to a new CSV file with updated labels (optional).
        update_hyperparameters : bool
            Whether to perform hyperparameter optimization (default False).
        """
        if models == []:
            models = self._models.keys()

        data = self._combined_datadb.copy()
        if update_labels:
            data = self.combine_resource_data(pd.read_csv(update_labels))

        for model_key in models:
            if model_key not in self._models:
                print(f"Ignoring model key {model_key}")
                continue
            if model_key in flood_class_from_postcode_methods:
                self.fit_risk_label(data, model_key, update_hyperparameters)
            elif model_key in flood_class_from_location_methods:
                self.fit_risk_label(data, model_key, update_hyperparameters)
            elif model_key in house_price_methods:
                self.fit_median_houseprice(
                    data, model_key, update_hyperparameters
                )
            elif model_key in historic_flooding_methods:
                self.fit_historic_flooding(
                    data, model_key, update_hyperparameters
                )
            elif model_key in local_authority_methods:
                self.fit_local_authority(
                    data, model_key, update_hyperparameters
                )

    def fit_local_authority(self, X, model_key, update_hyperparameters=False):
        data = X.copy()
        y = data["localAuthority"]
        X = data[["easting", "northing"]]
        print(f"Fitting model {model_key} for local authority estimation")
        m = self._models[model_key]
        estimator = m["estimator"]
        prep_pipe_factory = m.get("prep_pipe_factory", None)
        prep_pipe = None
        if prep_pipe_factory is not None:
            prep_pipe = prep_pipe_factory(X)
            print("PREP PIPE available, fitting...")
            X = prep_pipe.fit_transform(X)
        if update_hyperparameters and "sklearn" in str(type(estimator)):
            random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=m["param_grid"],
                cv=5,
                scoring="accuracy",
            )
            random_search.fit(X, y)
            estimator = random_search.best_estimator_
            print(
                f"Best hyperparameters for {model_key}: {random_search.best_params_}"
            )
        else:
            estimator.fit(X, y)
        m["estimator"] = estimator
        m["prep_pipe"] = prep_pipe

    def fit_median_houseprice(
        self, X, model_key, update_hyperparameters=False
    ):
        """
        This function fits models to predict median house prices.
        Note that rows with missing values in the target column are dropped.
        Also only a fraction of the data is used to train the model (5%) in order to speed up training.

        Parameters:
            X (pd.DataFrame): The input data
            models (list): A list of models to fit. The models should be in the keys of self._models
            update_hyperparameters (bool): If True, the hyperparameters of the models are updated using RandomizedSearchCV

        Returns:
            None
        """
        y_col = "medianPrice"
        # Drop rows with missing values in the target column
        X = X.dropna(subset=[y_col])

        # TODO: TO REMOVE
        # sample fraction
        sample_frac = 1
        X = X.sample(frac=sample_frac, random_state=42)

        # TODO: remove outliers
        X = X[X[y_col] < 1e6]

        y = X[y_col]
        # This subselection was determined after observing colinearity/duplication of information of these three features amongst the rest of the dataset.
        X = X[
            ["easting", "northing", "postcodeDistrict"]
        ]  # derivatives of postcode

        print(f"Fitting model {model_key} for median house price")
        m = self._models[model_key]
        estimator = m["estimator"]
        # Retrieving the relevant pipeline for data processing
        prep_pipe_factory = m.get("prep_pipe_factory", None)
        if prep_pipe_factory is not None:
            prep_pipe = prep_pipe_factory(X)
            X = prep_pipe.fit_transform(X)
            m["prep_pipe"] = prep_pipe

        if update_hyperparameters:
            random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=m["param_grid"],
                cv=5,
                scoring="neg_mean_absolute_error",  # Use MAE because of outliers ??? but in scoring we are using RMSE which is very sensitive to outliers????
            )
            random_search.fit(X, y)
            estimator = random_search.best_estimator_
            print(
                f"Best hyperparameters for {model_key}: {random_search.best_params_}"
            )
        else:
            estimator.fit(X, y)

        # Print R2, RMSE, MAE
        y_pred = estimator.predict(X)
        # Note that this formatting of calculating R2 may be different from what is commonly done with sklearn r2_score()
        r2 = estimator.score(X, y)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mae = np.mean(np.abs(y - y_pred))
        print(f"R2: {r2}, RMSE: {rmse}, MAE: {mae}")

        m["estimator"] = estimator
        m["prep_pipe"] = prep_pipe

    def fit_historic_flooding(
        self, X, model_key, update_hyperparameters=False
    ):
        """
        This function fits models to predict historic flooding.
        Note that rows with missing values in the target column are dropped.

        Parameters:
            X (pd.DataFrame): The input data
            models (list): A list of models to fit. The models should be in the keys of self._models
            update_hyperparameters (bool): If True, the hyperparameters of the models are updated using RandomizedSearchCV

        Returns:
            None
        """
        y_col = "historicallyFlooded"
        # Drop rows with missing values in the target column
        X = X.dropna(subset=[y_col])

        # undersample the majority class to balance the dataset
        flooded = X[X[y_col] == True]
        not_flooded = X[X[y_col] == False]
        n_flooded = len(flooded)
        n_not_flooded = len(not_flooded)
        if n_flooded > n_not_flooded:
            flooded = flooded.sample(n=n_not_flooded, random_state=42)
        else:
            not_flooded = not_flooded.sample(n=n_flooded, random_state=42)
        X = pd.concat([flooded, not_flooded])

        y = X[y_col]
        X = X[["easting", "northing", "postcodeDistrict"]]

        print(f"Fitting model {model_key} for historic flooding")
        m = self._models[model_key]
        estimator = m["estimator"]
        prep_pipe_factory = m.get("prep_pipe_factory", None)
        prep_pipe = None
        if prep_pipe_factory is not None:
            prep_pipe = prep_pipe_factory(X)
            X = prep_pipe.fit_transform(X)

        if update_hyperparameters:
            random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=m["param_grid"],
                cv=5,
                scoring="f1",
            )
            random_search.fit(X, y)
            estimator = random_search.best_estimator_
            print(
                f"Best hyperparameters for {model_key}: {random_search.best_params_}"
            )
        else:
            estimator.fit(X, y)
        m["estimator"] = estimator
        m["prep_pipe"] = prep_pipe

    def fit_risk_label(self, X, model_key, update_hyperparameters=False):
        """

        This function fits models to predict flood risk labels.
        Note that rows with missing values in the target column are dropped.

        Parameters:
            X (pd.DataFrame): The input data
            models (list): A list of models to fit. The models should be in the keys of self._models
            update_hyperparameters (bool): If True, the hyperparameters of the models are updated using RandomizedSearchCV

        Returns:
            None
        """
        data = X.copy()
        y = data["riskLabel"]
        data = data.dropna(subset=["riskLabel"])
        X = data[["easting", "northing", "soilType", "elevation"]]

        print(f"Fitting model {model_key} for riskLabel estimation")
        m = self._models[model_key]
        estimator = m["estimator"]
        prep_pipe_factory = m.get("prep_pipe_factory", None)
        prep_pipe = None
        X_tran = X
        if prep_pipe_factory is not None:
            prep_pipe = prep_pipe_factory(X)
            print("PREP PIPE available, fitting...")
            X_tran = prep_pipe.fit_transform(X)
        if update_hyperparameters and "sklearn" in str(type(estimator)):
            random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=m["param_grid"],
                cv=5,
                scoring="f1_weighted",
            )
            random_search.fit(X_tran, y)
            estimator = random_search.best_estimator_
            print(
                f"Best hyperparameters for {model_key}: {random_search.best_params_}"
            )
        else:
            estimator.fit(X_tran, y)
        m["estimator"] = estimator
        m["prep_pipe"] = prep_pipe

        # Compute predictions and custom scores
        y_pred = estimator.predict(X_tran)
        conf_matrix = confusion_matrix(y, y_pred)
        # Plot confusion matrix
        unique_labels = sorted(set(y))  # Ensure labels are sorted
        self._plot_confusion_matrix(conf_matrix, unique_labels, model_key)
        m["estimator"] = estimator

    def _plot_confusion_matrix(self, conf_matrix, labels, model_name):
        """
        Plotting confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
        )
        plt.xlabel("True Label", fontsize=12)
        plt.ylabel("Predicted Label", fontsize=12)
        plt.title(f"Confusion Matrix for {model_name}", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        # save_path = os.path.join(
        #     "../notebooks", f"{model_name}_confusion_matrix.png"
        # )

        # Save the figure
        plt.savefig(f"{model_name}_confusion_matrix.png")
        plt.close()  # Close the figure to free memory

    def extract_postcode_info(self, df: pd.DataFrame) -> pd.DataFrame:
        # Note that this code duplicates some of the functionality present in the combine_resource_data function
        # For future enhancements we would recommend finding someway to write a single function that processes both labelled and unlabelled data
        # Another area for enhancement would be to split this function into its component parts for processing some datasets, and then merging others.
        """
        Extracts postcode information from the given DataFrame and enriches it with
        additional data from sector and district databases.
         - The input DataFrame must contain a 'postcode' column.
         - The method assumes that `self._sectordb` and `self._districtdb` are
              available and contain the necessary sector and district data.
         - The method performs several string manipulations and merges to
              enrich the input DataFrame.
         - Highly correlated columns are dropped before returning the final DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing a 'postcode' column.

        Returns:
            pd.DataFrame: DataFrame with enriched postcode information, including
            sector and district data, and calculated columns for sector headcount,
            households, and animal-to-human ratios.
        """
        # Extract postcodes information
        df["postcodeDistrict"] = df["postcode"].str.split(" ").str[0]
        df["postcodeArea"] = df["postcodeDistrict"].str.extract(
            r"([a-zA-Z]+)", expand=False
        )
        df["postcodeSector"] = (
            df["postcode"].str.split(" ").str[0]
            + " "
            + df["postcode"].str.split(" ").str[1].str[0]
        )
        # df = df.drop(columns=["postcode"]) dont drop else lookup will fial

        # Read and join sector and district data
        sector_data = self._sectordb
        sector_data["postcodeSector"] = sector_data[
            "postcodeSector"
        ].str.replace("   ", "  ")
        sector_data["postcodeSector"] = sector_data[
            "postcodeSector"
        ].str.replace("  ", " ")

        district_data = self._districtdb
        df = df.merge(sector_data, on="postcodeSector", how="left")
        df = df.merge(district_data, on="postcodeDistrict", how="left")

        df.rename(
            columns={
                "headcount": "sectorHeadcount",
                "households": "sectorHouseholds",
            },
            inplace=True,
        )
        df["disctrictCats"] = df["sectorHouseholds"] * df["catsPerHousehold"]
        df["disctrictDogs"] = df["sectorHouseholds"] * df["dogsPerHousehold"]
        df["sectorHeadPerHousehold"] = (
            df["sectorHeadcount"] / df["sectorHouseholds"]
        )
        df["sectorAnimalToHumanRatio"] = (
            df["catsPerHousehold"] + df["dogsPerHousehold"]
        ) / df["sectorHeadPerHousehold"]
        df.sort_values(by="postcodeDistrict", inplace=True)

        # Drop highly correlated columns
        df = df.drop(
            columns=[
                "catsPerHousehold",
                "dogsPerHousehold",
                "disctrictCats",
                "disctrictDogs",
            ]
        )
        return df

    def lookup_easting_northing(
        self, postcodes: Sequence, dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """Get a dataframe of OS eastings and northings from a sequence of
        input postcodes in the labelled or unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the easting and northing columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of 'easthing' and 'northing',
            indexed by the input postcodes. Invalid postcodes (i.e. those
            not in the available postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['RH16 2QE'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        >>> results = tool.lookup_easting_northing(['RH16 2QE', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        AB1 2PQ        NaN       NaN
        """
        df = self._lookupdb.copy()
        df = df.set_index("postcode")
        df = df.reindex(postcodes)
        # remove the 'postcode' from the index
        df.index.name = None
        return df[["easting", "northing"]].astype(dtype)

    # Note that this function is actually never used in our implementation of the project, but may be useful for future uses.
    def lookup_lat_long(
        self, postcodes: Sequence, dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """
        Get a Pandas dataframe containing GPS latitude and longitude
        information for a sequence of postcodes in the labelled or
        unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the latitude and longitude columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e. those not in
            the input unlabelled postcodes file) return as NaNs in the latitude
            and longitude columns.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL']) # doctest: +SKIP
                latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        """
        # Get the easting and northing data
        easting_northing_dataframe = self.lookup_easting_northing(
            postcodes, dtype
        )
        # Get the latitude and longitude data
        lat_df, long_df = get_gps_lat_long_from_easting_northing(
            easting_northing_dataframe["easting"],
            easting_northing_dataframe["northing"],
            rads=False,
            dms=False,
        )
        # Create a DataFrame from the results & ensure index matches input postcodes
        lat_long_dataframe = pd.DataFrame(
            {"latitude": lat_df, "longitude": long_df},
            index=easting_northing_dataframe.index,
        )

        # Align the result with the input postcodes (preserve order and handle missing postcodes)
        result = lat_long_dataframe.reindex(postcodes)

        return result

    def impute_missing_values(
        self,
        dataframe: pd.DataFrame,
        method: str = None,
        constant_values: dict = IMPUTATION_CONSTANTS,
    ) -> pd.DataFrame:
        """Impute missing values in a dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame potentially containing missing values (NaNs) or with missing columns.
        method : str, optional
            Method to use for imputation. Options include:
            - 'mean', to use the mean for the numerical columns.
            - 'knn' to use k-nearest neighbors imputation for numerical columns.
            - 'constant' to use the constant method for the columns available.
        constant_values : dict, optional
            Dictionary containing constant values to use for imputation in the format {column_name: value}.
            Only used if method is not specified or 'constant' method is used.

        Returns
        -------
        pandas.DataFrame
            DataFrame with missing values imputed.
        """

        # Columns to exclude from imputation
        exclude_numeric_list = ["easting", "northing"]
        exclude_categorical_list = ["postcode"]

        # Safely separate excluded columns if they exist
        # excluded_numeric = dataframe[exclude_numeric] if set(exclude_numeric).issubset(dataframe.columns) else pd.DataFrame()
        # excluded_categorical = dataframe[exclude_categorical] if set(exclude_categorical).issubset(dataframe.columns) else pd.DataFrame()
        excluded_numeric = (
            dataframe[exclude_numeric_list]
            if all(col in dataframe.columns for col in exclude_numeric_list)
            else pd.DataFrame()
        )
        excluded_categorical = (
            dataframe[exclude_categorical_list]
            if all(
                col in dataframe.columns for col in exclude_categorical_list
            )
            else pd.DataFrame()
        )

        # Identify numeric and categorical columns for imputation
        numeric_columns = dataframe.select_dtypes(
            include=[np.number]
        ).columns.difference(exclude_numeric_list)
        categorical_columns = dataframe.select_dtypes(
            include=[object]
        ).columns.difference(exclude_categorical_list)

        # Handle imputation for numeric columns
        if method == "mean" and not numeric_columns.empty:
            imputer = SimpleImputer(strategy="mean")
            dataframe[numeric_columns] = imputer.fit_transform(
                dataframe[numeric_columns]
            )

        elif (
            method == "knn"
            and not numeric_columns.empty
            and dataframe[numeric_columns].isnull().any().any()
        ):
            imputer = KNNImputer(n_neighbors=5)
            dataframe[numeric_columns] = imputer.fit_transform(
                dataframe[numeric_columns]
            )

        elif method is None:
            for column in numeric_columns:
                if column in constant_values:
                    dataframe[column] = dataframe[column].fillna(
                        constant_values[column]
                    )

        # Re-check numeric columns for any remaining NaNs
        if not numeric_columns.empty:
            for column in numeric_columns:
                if dataframe[column].isnull().any():
                    dataframe[column] = dataframe[column].fillna(
                        dataframe[column].mean()
                    )

        # Handle imputation for categorical columns
        if not categorical_columns.empty:
            for column in categorical_columns:
                if column in constant_values:
                    dataframe[column] = dataframe[column].fillna(
                        constant_values[column]
                    )
                else:
                    # Use mode for categorical columns if no specific constant provided
                    dataframe[column] = dataframe[column].fillna(
                        dataframe[column].mode()[0]
                    )

        # Reinsert excluded columns if they were separated
        if not excluded_numeric.empty:
            dataframe[exclude_numeric_list] = excluded_numeric
        if not excluded_categorical.empty:
            dataframe[exclude_categorical_list] = excluded_categorical

        return dataframe

    # code below this line has not been checked by Lemuel

    def predict_flood_class_from_postcode(
        self, postcodes: Sequence[str], method: str = "all_zero_risk"
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
            Returns NaN for postcode units not in the available postcode files.
        """
        print(f"Predicting flood classes from {len(postcodes)} postcodes")
        if method == "all_zero_risk":
            return pd.Series(
                data=np.ones(len(postcodes), int),
                index=np.asarray(postcodes),
                name="riskLabel",
            )
        else:
            # build the feature set from the postcodes
            df = self._lookupdb.copy()
            X = df[df["postcode"].isin(postcodes)]
            print(f"Found {len(X)} postcodes in the lookup database")
            assert not X.empty, "No postcodes found in the lookup database"
            X = X[["easting", "northing", "soilType", "elevation"]]

            prep_pipe = self._models[method].get("prep_pipe", None)
            if prep_pipe is not None:
                X = prep_pipe.transform(X)

            y_pred = self._models[method]["estimator"].predict(X)
            return pd.Series(
                data=y_pred, index=np.asarray(postcodes), name="riskLabel"
            )

    def predict_flood_class_from_OSGB36_location(
        self,
        eastings: Sequence[float],
        northings: Sequence[float],
        method: str = "all_zero_risk",
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        """

        if method == "all_zero_risk":
            return pd.Series(
                data=np.ones(len(eastings), int),
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name="riskLabel",
            )
        else:
            assert len(eastings) == len(
                northings
            ), "eastings and northings must be the same length"
            print(
                f"Predicting flood classes from {len(eastings)} OSGB36 locations"
            )
            # build feature set df
            X = pd.DataFrame(
                {
                    "easting": eastings,
                    "northing": northings,
                }
            )
            # fill in the corresponding soil type and elevation from the lookup using the eastings and northings
            X = X.merge(self._lookupdb, on=["easting", "northing"], how="left")
            X = X.drop_duplicates(subset=["easting", "northing"])
            assert len(X) == len(eastings)

            X = X[["easting", "northing", "soilType", "elevation"]]

            prep_pipe = self._models[method].get("prep_pipe", None)
            if prep_pipe is not None:
                X = prep_pipe.transform(X)

            y_pred = self._models[method]["estimator"].predict(X)
            return pd.Series(
                data=y_pred,
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name="riskLabel",
            )

    def predict_flood_class_from_WGS84_locations(
        self,
        longitudes: Sequence[float],
        latitudes: Sequence[float],
        method: str = "all_zero_risk",
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : str (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels multi-indexed by
            location as a (longitude, latitude) pair.
        """

        if method == "all_zero_risk":
            idx = pd.MultiIndex.from_tuples(
                [(lng, lat) for lng, lat in zip(longitudes, latitudes)]
            )
            return pd.Series(
                data=np.ones(len(longitudes), int),
                index=idx,
                name="riskLabel",
            )
        else:
            df = self._combined_datadb.copy(deep=True)
            X = df[
                df["longitude"].isin(longitudes)
                & df["latitude"].isin(latitudes)
            ]
            # If no postcodes are found, return all ones
            if X.empty:
                idx = pd.MultiIndex.from_tuples(
                    [(lng, lat) for lng, lat in zip(longitudes, latitudes)]
                )
                return pd.Series(
                    data=np.ones(len(longitudes), int),
                    index=idx,
                    name="riskLabel",
                )
            X_pred = self.impute_missing_values(X)
            X_pred_df = pd.DataFrame(
                X_pred,
                columns=[
                    # "postcode",
                    "elevation",
                    # "distanceToWatercourse",
                    "soilType",
                    # "nearestWatercourse",
                    # "localAuthority",
                    "easting",
                    "northing",
                ],
            )
            X_pred_tran = X_pred_df
            if self._models[method].get("prep_pipe", None) is not None:
                X_pred_tran = self._models[method]["prep_pipe"].transform(
                    X_pred_df
                )
            y_pred = self._models[method]["estimator"].predict(X_pred_tran)
            location_tuples = [
                (lng, lat) for lng, lat in zip(longitudes, latitudes)
            ]

            # Use this location_tuples as the index of the Series
            return pd.Series(
                data=y_pred, index=location_tuples, name="riskLabel"
            )

    def predict_median_house_price(
        self, postcodes: Sequence[str], method: str = "all_england_median"
    ) -> pd.Series:
        """
        Generate series predicting median house price for a collection
        of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """

        if method == "all_england_median":
            return pd.Series(
                data=np.full(len(postcodes), 245000.0),
                index=np.asarray(postcodes),
                name="medianPrice",
            )
        else:
            m = self._models[method]
            estimator = m["estimator"]

            df = pd.DataFrame(
                {
                    "postcode": postcodes,
                }
            )
            df = df.merge(self._lookupdb, on="postcode", how="left")
            df = df.drop_duplicates(subset=["postcode"])
            X = df[["easting", "northing", "postcodeDistrict"]]
            prep_pipe = m.get("prep_pipe", None)
            if prep_pipe is not None:
                X = prep_pipe.transform(X)

            y_pred = estimator.predict(X)
            return pd.Series(
                data=y_pred, index=postcodes, name="medianPrice"
            )

    def predict_local_authority(
        self,
        eastings: Sequence[float],
        northings: Sequence[float],
        method: str = "do_nothing",
    ) -> pd.Series:
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            locations, and multiindexed by the location as a
            (easting, northing) tuple.
        """
        idx = pd.MultiIndex.from_tuples(
            [(est, nth) for est, nth in zip(eastings, northings)],
            names=["easting", "northing"],
        )
        if method == "all_nan":
            return pd.Series(
                data=np.full(len(eastings), np.nan),
                index=idx,
                name="localAuthority",
            )
        else:
            # Create DataFrame X from eastings and northings
            X = pd.DataFrame({"easting": eastings, "northing": northings})

            # Merge X with lookupdb to get the postcodeDistrict
            X = X.merge(self._lookupdb, on=["easting", "northing"], how="left")
            X = X.drop_duplicates(subset=["easting", "northing"])
            X = X[["easting", "northing"]]

            prep_pipe = self._models[method].get("prep_pipe", None)
            if prep_pipe is not None:
                X = prep_pipe.transform(X)

            y_pred = self._models[method]["estimator"].predict(X)
            return pd.Series(data=y_pred, index=idx, name="localAuthority")

    def predict_historic_flooding(
        self, postcodes: Sequence[str], method: str = "all_false"
    ) -> pd.Series:
        """
        Generate series predicting whether a collection of postcodes
        has experienced historic flooding.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        """

        if method == "all_false":
            return pd.Series(
                data=np.full(len(postcodes), False),
                index=np.asarray(postcodes),
                name="historicallyFlooded",
            )
        else:
            df = self._lookupdb.copy()
            X = df[df["postcode"].isin(postcodes)]
            print(f"Found {len(X)} postcodes in the lookup database")
            assert not X.empty, "No postcodes found in the lookup database"
            X = X[["easting", "northing", "postcodeDistrict"]]

            prep_pipe = self._models[method].get("prep_pipe", None)
            if prep_pipe is not None:
                X = prep_pipe.transform(X)

            y_pred = self._models[method]["estimator"].predict(X)
            # assert all elements in y_pred are either 1 or 0
            assert all(
                [y == 1 or y == 0 for y in y_pred]
            ), "y_pred contains values other than 1 or 0"
            print(y_pred)
            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes),
                name="historicallyFlooded",
            )

    def estimate_total_value(self, postal_data: Sequence[str]) -> pd.Series:
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        The estimate is based on the median house price for the area and an
        estimate of the number of properties it contains.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcode sectors (either
            may be used).


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """
        postcodes = postal_data
        median_price = self.predict_median_house_price(postcodes)
        # Get the number of houses per postcode
        sector_data = pd.read_csv("../flood_tool/resources/sector_data.csv")
        postcode_to_households = sector_data.set_index("postcodeSector")[
            "households"
        ]
        # Identification postcode
        households_list = []
        for postcode in postcodes:
            # Split the postcodes into two parts by space
            parts = postcode.split(" ")
            if len(parts) > 1:
                if len(parts[0]) == 4:
                    sector_prefix = f"{parts[0]} {parts[1][0]}"
                elif len(parts[0]) == 3:
                    sector_prefix = f"{parts[0]}  {parts[1][0]}"
                else:
                    sector_prefix = f"{parts[0]}   {parts[1][0]}"
                if sector_prefix in postcode_to_households:
                    households_list.append(
                        postcode_to_households[sector_prefix]
                    )
                else:
                    # If no match is found, the population is 0
                    households_list.append(0)
            else:
                households_list.append(0)
        households = pd.Series(households_list, index=postcodes)
        # Caculate
        total_value = median_price * households
        return pd.Series(
            total_value, index=postcodes, name="TotalPropertyValue"
        )

    def estimate_annual_human_flood_risk(
        self, postcodes: Sequence[str], risk_labels: [pd.Series | None] = None
    ) -> pd.Series:
        """
        Return a series of estimates of the risk to human life for a
        collection of postcodes.

        Risk is defined here as an impact coefficient multiplied by the
        estimated number of people under threat multiplied by the probability
        of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual human flood risk estimates
            indexed by postcode.
        """

        risk_labels = risk_labels or self.predict_flood_class_from_postcode(
            postcodes
        )
        sector_data = pd.read_csv("../flood_tool/resources/sector_data.csv")
        # Calculate the average population of each postcode
        sector_data["population_per_unit"] = (
            sector_data["headcount"] / sector_data["numberOfPostcodeUnits"]
        )
        postcode_to_population_mapping = sector_data.set_index(
            "postcodeSector"
        )["population_per_unit"]

        # Identification postcode
        population_per_postcode = []
        for postcode in postcodes:
            # Split the postcodes into two parts by space
            parts = postcode.split(" ")
            if len(parts) > 1:
                if len(parts[0]) == 4:
                    sector_prefix = f"{parts[0]} {parts[1][0]}"
                elif len(parts[0]) == 3:
                    sector_prefix = f"{parts[0]}  {parts[1][0]}"
                else:
                    sector_prefix = f"{parts[0]}   {parts[1][0]}"
                if sector_prefix in postcode_to_population_mapping:
                    population_per_postcode.append(
                        postcode_to_population_mapping[sector_prefix]
                    )
                else:
                    # If no match is found, the population is 0
                    population_per_postcode.append(0)  # Very important***
            else:
                # If no matching partition is found, the default population is 0
                population_per_postcode.append(0)

        postcode_to_population = pd.Series(
            population_per_postcode, index=postcodes
        )

        # Risk probability mapping
        risk_probability_map = {
            1: 0.001,
            2: 0.02,
            3: 0.005,
            4: 0.01,
            5: 0.02,
            6: 0.03,
            7: 0.05,
        }
        risk_probabilities = risk_labels.map(risk_probability_map)

        # Caculate
        human_coefficient = 0.1
        human_risk = (
            human_coefficient
            * postcode_to_population.loc[postcodes]
            * risk_probabilities
        )
        # Replace 0 values with NaN
        human_risk.replace(0, np.nan, inplace=True)

        return pd.Series(
            human_risk, index=postcodes, name="AnnualHumanFloodRisk"
        )

    def estimate_annual_flood_economic_risk(
        self, postcodes: Sequence[str], risk_labels: [pd.Series | None] = None
    ) -> pd.Series:
        """
        Return a series of estimates of the total economic property risk
        for a collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            optionally provide a Pandas Series containing flood risk
            classifiers, as predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual economic flood risk estimates indexed
            by postcode.
        """

        risk_labels = risk_labels or self.predict_flood_class_from_postcode(
            postcodes
        )
        total_property_values = self.estimate_total_value(postcodes)
        # Risk probability mapping
        risk_probability_map = {
            1: 0.001,
            2: 0.02,
            3: 0.005,
            4: 0.01,
            5: 0.02,
            6: 0.03,
            7: 0.05,
        }
        risk_probabilities = risk_labels.map(risk_probability_map)
        # Caculate
        economic_coefficient = 0.05
        economic_risk = (
            economic_coefficient * total_property_values * risk_probabilities
        )
        # Replace 0 values with NaN
        economic_risk.replace(0, np.nan, inplace=True)

        return pd.Series(
            economic_risk, index=postcodes, name="AnnualEconomicFloodRisk"
        )

    ## Daniel's Data Cleaning / Processing Functions
    def clean_postcode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and processes postcode data in the given DataFrame.
        This function extracts and cleans various components of the postcode, including the incode, outcode, area, and sector.
        It also calculates the latitude and longitude based on easting and northing coordinates.

        Parameters:
            df (pd.DataFrame): DataFrame containing a 'postcode' column and easting/northing coordinates.
        Returns:
            pd.DataFrame: DataFrame with additional columns for incode, outcode, area, sector, latitude, and longitude.
        """
        df["incode"] = (
            df["postcode"]
            .astype(str)
            .str[-3:]
            .str.replace(r"[\W_]", "", regex=True)
        )
        df["postcodeDistrict"] = df["postcode"].str.split(" ").str[0]
        df["outcode"] = (
            df["postcode"]
            .astype(str)
            .str[:-3]
            .str.replace(r"[\W_]", "", regex=True)
        )
        df["area"] = (
            df["outcode"]
            .astype(str)
            .str[:2]
            .str.replace(r"[\W_]", "", regex=True)
        )
        df["sector"] = (df["outcode"] + " " + df["incode"].str[:1]).astype(str)
        lat, long = get_gps_lat_long_from_easting_northing(
            df["easting"], df["northing"]
        )
        df["latitude"] = lat
        df["longitude"] = long
        return df

    def clean_sector_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and processes sector data in the given DataFrame.
        This function extracts and cleans various components of the sector, including the incode, outcode, area, and sector.
        It also calculates the households per incode unit and headcount per household.

        Parameters:
            df (pd.DataFrame): DataFrame containing sector data.
        Returns:
            pd.DataFrame: DataFrame with additional columns for households per incode unit, headcount per household, outcode, and incode first value.
        """
        df["households_per_incode_unit"] = (
            df["households"] / df["numberOfPostcodeUnits"]
        )
        df["headcount_per_household"] = df["headcount"] / df["households"]
        df["outcode"] = df["postcodeSector"].str.split(" ").str[0]
        df["incode_first_val"] = df["postcodeSector"].str[-1:]
        df["postcodeSector_match"] = (
            df["outcode"] + " " + df["incode_first_val"]
        ).astype(str)
        return df

    def clean_station_data(self, df: pd.DataFrame) -> pd.DataFrame:
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in cast"
        )
        """
        Cleans the station data by calculating and adding easting and northing values.
        This function takes a DataFrame containing station data with latitude and longitude columns, calculates the corresponding easting and northing values, and adds these values as new columns to the DataFrame.
        
        Parameters:
            df (pd.DataFrame): A DataFrame containing station data with 'latitude' and 'longitude' columns.
        Returns:
            pd.DataFrame: The input DataFrame with additional 'easting' and 'northing' columns.
        """
        # calculating easting and northing values from latitude and longitude
        easting, northing = get_easting_northing_from_gps_lat_long(
            df["latitude"], df["longitude"]
        )

        # adding easting and northing values to the stations_data dataframe
        df["easting"] = easting.astype(int)
        df["northing"] = northing.astype(int)
        return df

    def merge_postcode_and_stations_data(
        self,
        postcodes_df: pd.DataFrame,
        stations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges the station and postcode data based on the closest station to each postcode.
        This function calculates the Euclidean distance between each postcode and each station, finds the closest station to each postcode, and merges the station and postcode data based on the closest station index.

        Parameters:
            stations_df (pd.DataFrame): DataFrame containing station data.
            postcodes_df (pd.DataFrame): DataFrame containing labelled postcode data.
        Returns:
            pd.DataFrame: Merged DataFrame containing station and postcode data.
        """
        # Calculating the euclidean distance between each postcode and each station
        euc_distance_to_stations = euclidean_distance(
            stations_df["stations_easting"],
            stations_df["stations_northing"],
            postcodes_df["easting"],
            postcodes_df["northing"],
        )

        # Finding the closest station to each postcode
        closest_postcode_distances = np.min(euc_distance_to_stations, axis=1)
        closest_station_indices = np.argmin(euc_distance_to_stations, axis=1)

        # Adding the closest station index and distance to the postcodes_labelled_data dataframe
        postcodes_df["closest_station_index"] = closest_station_indices
        postcodes_df["closest_station_distance"] = closest_postcode_distances

        # Resetting stations data index
        stations_df = stations_df.reset_index(drop=True)

        # Merging the postcodes_labelled_data with the stations_data_reset to get the closest station information
        merged_data = pd.DataFrame(
            postcodes_df.merge(
                stations_df,
                left_on="closest_station_index",
                right_on=stations_df.index,
                how="left",
            )
        )
        return merged_data

    def combine_resource_data(
        self,
        district_data: pd.DataFrame,
        postcode_data: pd.DataFrame,
        sector_data: pd.DataFrame,
        station_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combines the postcode, district, sector, and station data into a single DataFrame.

        Parameters:
            None (uses the class attributes for postcode, district, sector, and station data)

        Returns:
            pd.DataFrame: DataFrame containing combined postcode, district, sector, and station

        """
        # Cleaning the data
        district_data = district_data
        postcode_data = self.clean_postcode_data(postcode_data)
        sector_data = self.clean_sector_data(sector_data)
        station_data = self.clean_station_data(station_data)

        # Renaming the columns
        district_data = district_data.rename(
            columns=lambda col: "district_" + col
        )
        sector_data = sector_data.rename(columns=lambda col: "sector_" + col)
        station_data = station_data.rename(
            columns=lambda col: "stations_" + col
        )

        # Merging the data
        df = self.merge_postcode_and_stations_data(postcode_data, station_data)
        df = df.merge(
            district_data,
            left_on="outcode",
            right_on="district_postcodeDistrict",
            how="left",
        )
        df = df.merge(
            sector_data,
            left_on="sector",
            right_on="sector_postcodeSector_match",
            how="left",
        )

        df = df.drop(
            columns=[
                "district_postcodeDistrict",
                "sector_postcodeSector",
                "sector_outcode",
                "sector_incode_first_val",
                "sector_postcodeSector_match",
            ],
            axis=1,
        )
        df = df.rename(
            columns={
                "sector_households_per_incode_unit": "households_per_postcode",
                "sector_headcount_per_household": "headcount_per_household",
            }
        )
        return df

    def validate_postcode_input(postcode):
        """
        Validate the input postcode data and convert it to a pandas.Series if possible and needed

        Parameters:
            postcode (pandas.Series, list, tuple, numpy.ndarray, pandas.Index, str): Input postcode data to validate

        Returns:
            pandas.Series: Validated postcode data as a pandas.Series
            or
            ValueError: If the input is not a valid type

        """
        if isinstance(postcode, pd.Series):
            return postcode
        elif isinstance(postcode, (list, tuple, np.ndarray, pd.Index)):
            return pd.Series(postcode)
        elif isinstance(postcode, str):
            return pd.Series([postcode])
        else:
            raise ValueError(
                "Input must be a pandas.Series, list, tuple, numpy.ndarray, pandas.Index, or string"
            )

    def standardize_postcode(postcode):
        """
        Standardize the postcode format by removing whitespace and converting to uppercase
        and formatting the postcode to have a space before the last 3 characters.
        Also removes any non-alphanumeric characters from the postcode

        Parameters:
            postcode (pandas.Series): Input postcode data to standardize

        Returns:
            pandas.Series: Standardized postcode data

        """
        # Remove whitespace or non-alphanumeric characters from the postcode and convert to uppercase
        postcode = postcode.str.replace(r"[^a-zA-Z0-9]", "", regex=True)
        postcode = postcode.str.upper()

        def format_postcode(pc):
            """
            Format the postcode to have a space before the last 3 characters
            Ensure the postcode is at least 5 characters long before formatting
            Fill missing characters with NaN

            Parameters:
                pc (str): Input postcode to format

            Returns:
                str: Formatted postcode with a space before the last 3 characters
            """

            if len(pc) >= 5 and len(pc) <= 7:
                return pc[:-3] + " " + pc[-3:]
            return np.nan

        # Apply the formatting function to each postcode
        return postcode.apply(format_postcode)
