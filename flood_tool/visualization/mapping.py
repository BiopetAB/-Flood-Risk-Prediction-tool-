import folium
import plotly.graph_objects as go
import pandas as pd

__all__ = ["plot_circle", "plot_house_price", "plot_feature_density"]


def plot_circle(lat, lon, radius, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> plot_circle(52.79, -2.95, 1e3, map=None) # doctest: +SKIP
    """

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle(
        location=[lat, lon],
        radius=radius,
        fill=True,
        fillOpacity=0.6,
        **kwargs,
    ).add_to(map)

    return map


# Daniel's function to plot house prices
def plot_house_price(basedf: pd.DataFrame, predicteddf: pd.DataFrame):
    """
    Plot the actual and predicted house prices on a map.

    Parameters
    ----------
    basedf: pd.DataFrame
        The input data
    
    predicteddf: pd.DataFrame
        The predicted data

    Returns
    -------
    None


    """
    fig = go.Figure()

    # Add actual prices
    fig.add_trace(
        go.Densitymapbox(
            lat=basedf["latitude"],
            lon=basedf["longitude"],
            z=basedf["medianPrice"],
            radius=10,
            colorscale="Blues",
            name="Actual Prices",
        )
    )

    fig.add_trace(go.Densitymapbox(
        lat=basedf['latitude'],
        lon=basedf['longitude'],
        z=predicteddf['predicted_price'],
        radius=10,
        colorscale='Reds',
        name='Predicted Prices'
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 54.5, "lon": -2},
        mapbox_zoom=5,
        title="Actual vs Predicted House Prices",
    )
    fig.show()


def plot_feature_density(df, feature: str, plot_title: str):
    """
    Plot density of specified feature on a map.

    Parameters
    ----------
    df: pd.DataFrame
        The input data

    features: list
        The list of features to plot

    Returns
    -------
    None

    """
    fig = go.Figure()
    fig.add_trace(
        go.Densitymapbox(
            lat=df["latitude"],
            lon=df["longitude"],
            z=df[feature],
            radius=10,
            colorscale="Blues",
            name=feature,
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 54.5, "lon": -2},
        mapbox_zoom=5,
        title=plot_title,
    )
    fig.show()
