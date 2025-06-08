#!/usr/bin/env python

from setuptools import setup

# Modify, update, and improve this file as necessary.

setup(
    name="Flood Tool",
    version="1.2",
    description="Flood Risk Analysis Tool",
    author="ACDS project Team Kennet",  # update this
    packages=["flood_tool"],  # DON'T install the scoring package
    license="MIT",
    python_requires=">=3.12",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "folium",
        "scikit-learn",
        "plotly",
        "geopandas",
        "seaborn",
    ],
)
