# **Airbus Capstone Project**

This project works with Airbus flight and fuel/engine system data for comprehensive analysis and pattern detection. The aim is to manage missing data, detect outliers, and conduct Principal Component Analysis (PCA) and clustering for efficient data analysis and feature extraction.

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

- **Data Imputation:** Compute averages to fill in missing data.
- **Outlier Detection:** Conduct anomaly detection to identify outliers in the dataset.
- **Dimensionality Reduction:** Utilize techniques like Principal Component Analysis (PCA) to reduce the number of variables and to find the reconstruction error.

Stay tuned for updates as the project progresses.

## **Project Contributors**

- Federico Canadas
- Scott Liechtenstein
- Giacomo Tirelli
- Niels van Meijel
- Moritz Zoepffel
