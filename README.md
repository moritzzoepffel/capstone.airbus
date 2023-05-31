# Airbus Capstone Project

## The data consisted of the following (including the column KEY in the dataset)

### A/C and flight data:

- Time, day, month, year → ONLY MSN 02
- UTC date/time → UTC_TIME
- MSN (A/C Name) → ​​MSN
- Flight number → Flight
- Flight phase\* → FLIGHT_PHASE_COUNT
- Altitude → FW_GEO_ALTITUDE
- Pitch and roll → ONLY MSN 02

### Fuel/Engine system data:

- Engine status (Running or not). → ONLY MSN 02
- Fuel flow (to each engine) → ONLY MSN 022
- Fuel used (by engines; Kg):
  - FUEL_USED_1 → (Engine 1)
  - FUEL_USED_2 → (Engine 2)
  - FUEL_USED_3 → (Engine 3)
  - FUEL_USED_4 → (Engine 4)
- Fuel on board (“FOB” ; Kg) → VALUE_FOB
- Fuel quantity per collector cell and surge tank volume (Kg):
  - VALUE_FUEL_QTY_CT → Centra Tank
  - VALUE_FUEL_QTY_FT1 → Feed Tank 1 (Engine 1)
  - VALUE_FUEL_QTY_FT2 → Feed Tank 2 (Engine 2)
  - VALUE_FUEL_QTY_FT3 → Feed Tank 3 (Engine 3)
  - VALUE_FUEL_QTY_FT4 → Feed Tank 4 (Engine 4)
  - VALUE_FUEL_QTY_LXT → Transfer Tank Left
  - VALUE_FUEL_QTY_RXT → Transfer Tank Right
- Pump status (On/Off, normally/abnormally, immersed/not immersed). → ONLY MSN 02
- Leak detection and leak flow. → ONLY MSN 02
- Fuel transfer mode. → ONLY MSN 02

### (Flight Phases):

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

Each file is a different plane.

Impute average in missing data.

Outlier detection searching for anomaly

Too many variable

Reconstruction error PCA
