"""Test flood tool."""

import numpy as np

# from pytest import mark

import flood_tool.tool as tool


test_tool = tool.Tool()


def test_lookup_easting_northing():
    """Check"""

    data = test_tool.lookup_easting_northing(["RH16 2QE"])

    assert len(data.index) == 1
    assert "RH16 2QE" in data.index

    assert np.isclose(data.loc["RH16 2QE", "easting"], 535295).all()
    assert np.isclose(data.loc["RH16 2QE", "northing"], 123643).all()


def test_lookup_easting_northing_mixed():
    # TR18 2JP,147247,30227
    data = test_tool.lookup_easting_northing(["TR18 2JP"])

    assert len(data.index) == 1
    assert "TR18 2JP" in data.index

    assert np.isclose(data.loc["TR18 2JP", "easting"], 147247).all()
    assert np.isclose(data.loc["TR18 2JP", "northing"], 30227).all()

    # SW17 6LD,528660,171696
    # TR16 5DE,171757,45048
    data = test_tool.lookup_easting_northing(["SW17 6LD", "TR16 5DE"])

    assert len(data.index) == 2
    assert "SW17 6LD" in data.index
    assert "TR16 5DE" in data.index

    assert np.isclose(data.loc["SW17 6LD", "easting"], 528660).all()
    assert np.isclose(data.loc["SW17 6LD", "northing"], 171696).all()

    assert np.isclose(data.loc["TR16 5DE", "easting"], 171757).all()
    assert np.isclose(data.loc["TR16 5DE", "northing"], 45048).all()


# @mark.xfail  # We expect this test to fail until we write some code for it.
def test_lookup_lat_long():
    """Check"""

    # data = test_tool.lookup_lat_long(["M34 7QL"])
    data = test_tool.lookup_lat_long(["RH16 2QE"])

    assert len(data.index) == 1
    assert "RH16 2QE" in data.index

    # Easting: 535295
    # Longitude: -0.073395
    # Northing: 123643
    # Latitude: 50.996285

    assert np.isclose(
        data.loc["RH16 2QE", "latitude"], 50.996285, rtol=1.0e-3
    ).all()
    assert np.isclose(
        data.loc["RH16 2QE", "longitude"], -0.073395, rtol=1.0e-3
    ).all()


# Convenience implementation to be able to run tests directly.
if __name__ == "__main__":
    test_lookup_easting_northing()
    test_lookup_easting_northing_mixed()
    test_lookup_lat_long()
