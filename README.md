# Earthquake Analysis Using Machine Learning

This project, completed as part of a Summer Research Internship at IIIT Bangalore (Jun-Jul 2024), focuses on using machine learning to analyze seismic data. The study specifically examines earthquake magnitude prediction, P-wave and S-wave arrival times, and strong motion duration prediction. This work aims to improve early-warning systems by predicting critical seismic features.

## Project Overview

### Objective
The goal of this project is to:
- Predict earthquake magnitude and arrival times of P-waves and S-waves.
- Determine the duration of strong motion to evaluate potential structural impact.
- Analyze the correlation between seismic events to understand interrelationships.

### Methodology
1. **Dataset**: Utilized the [STEAD](https://ieeexplore.ieee.org/document/8871127) dataset for training and testing, focusing on seismic events recorded between 2011-2018.
2. **Model**: Implemented a Random Forest Regressor from Scikit-Learn to predict magnitude and arrival times.
3. **Feature Engineering**: Selected 10 critical features from the STEAD dataset based on feature importance.
4. **Evaluation**: Model evaluation metrics, feature importances, and earthquake parameter predictions were generated.

### Correlation Analysis
A custom script (`Correlate.py`) was created to assess relationships between seismic events, examining:
- Temporal and spatial proximity.
- Magnitude-based stress transfer and aftershock potential.

## Key Results
- Earthquake magnitude prediction accuracy.
- P-wave and S-wave arrival times with calculated strong motion duration.
- Correlation score to indicate potential interaction between seismic events.

## References
1. [STEAD Dataset](https://ieeexplore.ieee.org/document/8871127)
2. Strong Motion Threshold Information: [Strong Motion Center](http://www.isesd.hi.is/ESD_Local/Documentation/documentation.htm)

## Acknowledgments
This project was supervised by Prof. B. Ashok at IIIT Bangalore.

## Contact
For more details, please contact me at [email@example.com].
