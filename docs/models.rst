====================================
Regression and Classification models
====================================
 
*Expand this section appropriately to describe the models you build*. **Any papers or AI tools used to assist you in creating your models should be described here and referenced appropriately.**
 
Risk Class Prediction Models
============================
 
All Zero Risk
-------------
 
This model assumes all unlabeled flood risk data is near zero risk (band 1, the modal class), which is the modal class for the data set, which is strongly unbalanced. This is a baseline model to compare against. While it achieve a moderately good accuracy, it is not a useful model for risk prediction, and has little skill.
 
RF Cls Risk Postcode
-------------------------
 
This model assumes flood risk predictions based on postcode are biased towards the most common risk class, due to the class imbalance in the dataset. While it achieves decent accuracy, it is not effective for predicting flood risk as it fails to distinguish between different risk levels, limiting its practical value.
 
Knn Cls Risk Postcode
-------------------------
 
This model uses k-Nearest Neighbors to predict flood risk based on postcode.
 
Rf Cls Risk Location
-------------------------
 
This model uses Random Forest to predict flood risk based on location data([eastings, northings], [longitudes, latitudes]).
 
Knn Cls Risk Location
-------------------------
 
This model uses k-Nearest Neighbors to predict flood risk based on location data([eastings, northings], [longitudes, latitudes]).
 
House Price Prediction Models
=============================
 
All England median
------------------
 
This model assumes all unlabeled house price data is the median house price for England. This is a baseline model to compare against. While it achieve a moderately good accuracy, it is not a useful model for risk prediction, and has little skill.
 
Rf Reg House Price
------------------
 
This model uses Random Forest to predict house prices based on historic data.
 
Historic Flooding Prediction Models
=============================
 
All False
---------
 
This model assumes all unlabeled data represents no flooding risk. It serves as a baseline for comparison. While it achieves moderately good accuracy, it is not a useful model for predicting flooding risk, as it fails to identify actual flooding events and has limited predictive value.
 
Rf Cls Historic
----------------
 
This model uses Random Forest to predict house prices based on historic data.
 
Local Authority Prediction Models
=================================
 
Rf Cls Local Authority
----------------------
 
This model uses Random Forest to predict local authority based on postcode.