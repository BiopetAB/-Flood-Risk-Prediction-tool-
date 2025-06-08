################################
The Team Kennet Flood Prediction Tool
################################

This package implements a flood risk prediction and visualization tool.

Installation Instructions
-------------------------

**1. Clone the repository**  

   Clone the repository from the following link:

   https://github.com/ese-ada-lovelace-2024/ads-deluge-kennet.git

**2. Navigate to the Directory**

   Use cd to navigate to the directory containing the setup.py file:

   cd /path/to/your/module

**3. Install the package**  

   Use pip install with the path to the setup.py file:

   pip install .

**4. Optional: Editable Installation**

   If you want to make changes to the code, you can install the package in editable mode:

   pip install -e .



Quick Usage guide
-----------------

**1. Import the Package and Initialize**

import flood_tool as ft

tool = ft.Tool()

**2. Train a Model**

For example,train the model for the "historicallyFlooded" task with hyperparameter tuning:

tool.fit(models=["rf_cls_historic"],task="historicallyFlooded",update_hyperparameters=True,)


**3. Some Notes**

Replace rf_cls_historic with other model keys if needed.

Make sure your data files are correctly set up in the default paths or specified during initialization.


Further Documentation
---------------------

.. toctree::
   :maxdepth: 2

   data_formats
   models
   coordinates
   visualization


Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members: