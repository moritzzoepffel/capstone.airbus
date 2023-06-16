# **Airbus Capstone Project**

This project works with Airbus flight and fuel/engine system data for comprehensive analysis and fuel leakage detection. The aim is to manage missing data, detect outliers, and conduct Principal Component Analysis (PCA) and clustering for efficient data analysis and feature extraction.

## **Data Overview**

Data is available for various flight and fuel/engine system parameters. Here are the details:

### **A/C and Flight Data**

- Time, day, month, year (exclusive to MSN 02)
- UTC date/time (**`UTC_TIME`**)
- MSN (A/C Name) (**`MSN`**)
- Flight number (**`Flight`**)
- Flight phase\* (**`FLIGHT_PHASE_COUNT`**)
- Altitude (**`FW_GEO_ALTITUDE`**)
- Pitch and roll (exclusive to MSN 02)

### **Fuel/Engine System Data**

- Engine status (Running or not) - exclusive to MSN 02
- Fuel flow to each engine - exclusive to MSN 022
- Fuel used by engines (Kg):
  - **`FUEL_USED_1`** (Engine 1)
  - **`FUEL_USED_2`** (Engine 2)
  - **`FUEL_USED_3`** (Engine 3)
  - **`FUEL_USED_4`** (Engine 4)
- Fuel on board (FOB; Kg) (**`VALUE_FOB`**)
- Fuel quantity per collector cell and surge tank volume (Kg):
  - **`VALUE_FUEL_QTY_CT`** (Centra Tank)
  - **`VALUE_FUEL_QTY_FT1`** (Feed Tank 1 - Engine 1)
  - **`VALUE_FUEL_QTY_FT2`** (Feed Tank 2 - Engine 2)
  - **`VALUE_FUEL_QTY_FT3`** (Feed Tank 3 - Engine 3)
  - **`VALUE_FUEL_QTY_FT4`** (Feed Tank 4 - Engine 4)
  - **`VALUE_FUEL_QTY_LXT`** (Transfer Tank Left)
  - **`VALUE_FUEL_QTY_RXT`** (Transfer Tank Right)
- Pump status (On/Off, normally/abnormally, immersed/not immersed) - exclusive to MSN 02
- Leak detection and leak flow - exclusive to MSN 02
- Fuel transfer mode - exclusive to MSN 02

Each file corresponds to a different plane.

### **Flight Phases**

1. Pre-flight
2. Engine Run
3. Take-Off 1
4. Take-Off 2
5. Take-Off 3
6. Climbing 1
7. Climbing 2
8. Cruise
9. Descent
10. Approach
11. Landing
12. Post-flight

## **Project Objective**

- **Data Imputation:** Fill missing data by interpolating and drop the data where imputation is not possible.

- **Outlier Detection:** Conduct anomaly detection to identify outliers in the dataset. This step will ensure the robustness of the following analytical steps.

- **Feature Engineering:** This process is aimed at creating new variables or transforming existing ones to better represent the underlying data patterns and structures. By creating meaningful derived features or transforming existing ones appropriately, we can significantly improve the interpretability and results of our analysis. We will follow careful feature selection to keep only the most relevant features in our models.

- **Dimensionality Reduction and Feature Extraction:** We plan to use PCA to manage the large number of variables in the dataset. PCA will help us to extract the most meaningful features and reduce the complexity of the dataset, thereby enhancing the computational efficiency of subsequent algorithms. The reconstruction error after PCA will be a valuable indicator of the quality of the dimensionality reduction process.

- **Clustering:** Post PCA, we plan to apply various clustering algorithms on the transformed data to find patterns and groupings. The objective is to identify any inherent groupings and segmentations in the data that could provide valuable insights about different flight and fuel leakage.

We will be comparing the performances of different clustering algorithms and their applicability to our data.

Stay tuned for updates as the project progresses.

## **Project Contributors**

- Federico Canadas
- Scott Liechtenstein
- Giacomo Tirelli
- Niels van Meijel
- Moritz Zoepffel
