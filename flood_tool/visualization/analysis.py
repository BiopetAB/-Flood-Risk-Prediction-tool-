# flake8: noqa: E501

import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

__all__ = ["plot_risk_map", "plot_postcode_density", "filter_and_aggregation"]

DEFAULT_POSTCODE_FILE = os.path.join(
    os.path.dirname(__file__), "resources", "postcodes_unlabelled.csv"
)


def plot_postcode_density(
    postcode_file=DEFAULT_POSTCODE_FILE,
    coordinate=["easting", "northing"],
    dx=1000,
):
    """Plot a postcode density map from a postcode file."""

    pdb = pd.read_csv(postcode_file)

    bbox = (
        pdb[coordinate[0]].min() - 0.5 * dx,
        pdb[coordinate[0]].max() + 0.5 * dx,
        pdb[coordinate[1]].min() - 0.5 * dx,
        pdb[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y in pdb[coordinate].values:
        Z[math.floor((x - bbox[0]) / dx), math.floor((y - bbox[2]) / dx)] += 1

    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, norm=matplotlib.colors.LogNorm()
    )
    plt.axis("equal")
    plt.colorbar()


def plot_risk_map(risk_data, coordinate=["easting", "northing"], dx=1000):
    """Plot a risk map."""

    bbox = (
        risk_data[coordinate[0]].min() - 0.5 * dx,
        risk_data[coordinate[0]].max() + 0.5 * dx,
        risk_data[coordinate[1]].min() - 0.5 * dx,
        risk_data[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y, val in risk_data[["easting", "northing", "risk"]].values:
        Z[
            math.floor((x - bbox[0]) / dx), math.floor((y - bbox[2]) / dx)
        ] += val

    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, norm=matplotlib.colors.LogNorm()
    )
    plt.axis("equal")
    plt.colorbar()

    for x, y, val in risk_data[["easting", "northing", "risk"]].values:
        i = math.floor((x - bbox[0]) / dx)
        j = math.floor((y - bbox[2]) / dx)
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  # Ensure indices are valid
            Z[i, j] += val


def filter_and_aggregation(
    data: pd.DataFrame,
    parameter: str,
    qualifier: str,
    aggregate_method: str = "sum",
) -> pd.DataFrame:
    """
    Filter the data based on the specified parameter and qualifier, and aggregate the data based on the specified method.
    Meant for usage with data similar to the 'wet_day' and 'typical_day' samples provided.

    Parameters:
    -----------
    data: pd.DataFrame
        The input data
    parameter: str
        The parameter to filter the data by
        A column name in the data
    qualifier: str
        The qualifier to filter the data by
        A column name in the data
    aggregate_method: str
        The method to aggregate the data by
        A method available in the pandas library

    Returns:
    --------
    pd.DataFrame
        The aggregated data
    """
    # Filtering Data
    filtered = data[(data['parameter'] == parameter) & (data['qualifier'] == qualifier)]
    # ensure 'value' column is present
    if 'value' not in filtered.columns:
        raise ValueError("'value' column is missing in the filtered data.")
    filtered['value'] = pd.to_numeric(filtered['value'], errors='coerce')
    
    # check if there are numeric columns
    numeric_columns = filtered.select_dtypes(include="number")
    
    if numeric_columns.empty:
        raise ValueError("No numeric columns available for aggregation.")
    
    # aggregate the data based on the specified method
    return (
        filtered.groupby('date')
        .aggregate({'value': aggregate_method})
        .reset_index()
    )