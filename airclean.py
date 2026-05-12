import pandas as pd
import random
import numpy as np
import tensorflow as tf


def introduce_flight_column(dataset, flight_number=0):
    """Assigns flight segment IDs and keeps all cruise-phase rows (FLIGHT_PHASE_COUNT==8)."""
    dataset_tmp = dataset.copy()
    dataset_tmp = dataset_tmp.dropna()

    for i in range(1, dataset_tmp.shape[0]):
        if dataset_tmp.index[i] - dataset_tmp.index[i - 1] > 1:
            flight_number += 1
        dataset_tmp.at[dataset_tmp.index[i], "FLIGHT"] = flight_number

    dataset_tmp["FLIGHT"] = dataset_tmp["FLIGHT"].fillna(0)
    dataset_tmp = dataset_tmp[dataset_tmp["FLIGHT_PHASE_COUNT"] == 8]
    return dataset_tmp


def create_leak(dataset, leak_flow, start_index=None, end_index=None, random_seed=42):
    """Injects a synthetic fuel leak into a random tank starting at start_index (default: midpoint)."""
    random.seed(random_seed)
    dataset_tmp = dataset.copy()

    FUEL_TANK_COLS = [
        "VALUE_FUEL_QTY_CT",
        "VALUE_FUEL_QTY_RXT",
        "VALUE_FUEL_QTY_LXT",
        "VALUE_FUEL_QTY_FT1",
        "VALUE_FUEL_QTY_FT2",
        "VALUE_FUEL_QTY_FT3",
        "VALUE_FUEL_QTY_FT4",
    ]

    while True:
        fuel_tank = random.choice(FUEL_TANK_COLS)
        if dataset_tmp[fuel_tank].max() >= leak_flow:
            break

    if start_index is None:
        start_index = len(dataset_tmp) // 2

    increment = 0
    dataset_tmp["label"] = False

    for index in dataset_tmp.iloc[start_index:].index:
        dataset_tmp.at[index, "VALUE_FOB"] -= leak_flow + increment
        dataset_tmp.at[index, fuel_tank] -= leak_flow + increment
        dataset_tmp.at[index, "label"] = True
        increment += leak_flow

    dataset_tmp["VALUE_FOB"] = dataset_tmp["VALUE_FOB"].clip(lower=0)
    dataset_tmp[fuel_tank] = dataset_tmp[fuel_tank].clip(lower=0)
    return dataset_tmp


def split_by_flight(dataset, test_size=0.2, random_state=42):
    """Splits dataset by flight IDs to prevent temporal leakage. Returns (train_df, test_df)."""
    flights = dataset["FLIGHT"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(flights)
    n_test = max(1, int(len(flights) * test_size))
    test_flights = flights[:n_test]
    train_flights = flights[n_test:]
    return (
        dataset[dataset["FLIGHT"].isin(train_flights)].copy(),
        dataset[dataset["FLIGHT"].isin(test_flights)].copy(),
    )


def loss_since_start(x):
    """Returns cumulative fuel loss since flight start (positive when fuel decreases)."""
    return x.iloc[0] - x


def define_new_features(dataset):
    """Creates derived fuel balance features and removes the pre-cruise ground phase."""
    dataset_tmp = dataset.copy()

    dataset_tmp["TOTAL_FUEL_USED"] = 0
    for i in range(1, 5):
        dataset_tmp["TOTAL_FUEL_USED"] += dataset_tmp["FUEL_USED_" + str(i)]

    dataset_tmp["VALUE_FOB_DIFF"] = dataset_tmp.groupby("FLIGHT")["VALUE_FOB"].diff()
    dataset_tmp["VALUE_FOB_DIFF"] = dataset_tmp["VALUE_FOB_DIFF"].fillna(0)

    dataset_tmp["TOTAL_FOB_BY_QTY"] = (
        dataset_tmp["VALUE_FUEL_QTY_CT"]
        + dataset_tmp["VALUE_FUEL_QTY_FT1"]
        + dataset_tmp["VALUE_FUEL_QTY_FT2"]
        + dataset_tmp["VALUE_FUEL_QTY_FT3"]
        + dataset_tmp["VALUE_FUEL_QTY_FT4"]
        + dataset_tmp["VALUE_FUEL_QTY_LXT"]
        + dataset_tmp["VALUE_FUEL_QTY_RXT"]
    )

    dataset_tmp["DELTA_VFOB_VS_VFOBQTY"] = (
        dataset_tmp["VALUE_FOB"] - dataset_tmp["TOTAL_FOB_BY_QTY"]
    )

    dataset_tmp["ALTITUDE_DIFF"] = dataset_tmp["FW_GEO_ALTITUDE"].diff().abs()

    dataset_tmp["VALUE_FOB_MISSING"] = dataset_tmp.groupby("FLIGHT")[
        "VALUE_FOB"
    ].transform(loss_since_start)

    dataset_tmp["VALUE_FOB_MISSING_BY_QTY"] = dataset_tmp.groupby("FLIGHT")[
        "TOTAL_FOB_BY_QTY"
    ].transform(loss_since_start)

    value_fob_0 = dataset_tmp["VALUE_FOB"].iloc[0]
    dataset_tmp["VALUE_FOB_BY_FUEL_USED"] = value_fob_0 - dataset_tmp["TOTAL_FUEL_USED"]

    for flight_num in dataset_tmp["FLIGHT"].unique():
        first_index = dataset_tmp[dataset_tmp["FLIGHT"] == flight_num].index[0]
        min_value_sum_fuel_used = dataset_tmp[dataset_tmp["FLIGHT"] == flight_num][
            "TOTAL_FUEL_USED"
        ].idxmin()
        dataset_tmp.drop(
            dataset_tmp.loc[first_index:min_value_sum_fuel_used].index, inplace=True
        )

    return dataset_tmp


def fit_outlier_bounds(dataset):
    """Computes 2.5/97.5 percentile bounds per float64 column. Call only on training data."""
    bounds = {}
    for col in dataset.select_dtypes(include="float64").columns:
        lower, upper = np.percentile(dataset[col].dropna(), [2.5, 97.5])
        bounds[col] = (lower, upper)
    return bounds


def apply_outlier_capping(dataset, bounds):
    """Clips column values to precomputed bounds without dropping any rows."""
    dataset_tmp = dataset.copy()
    for col, (lower, upper) in bounds.items():
        if col in dataset_tmp.columns:
            dataset_tmp[col] = dataset_tmp[col].clip(lower, upper)
    return dataset_tmp


def add_features(dataset):
    """Adds TOTAL_FUEL_USED and TOTAL_FOB_BY_QTY aggregation columns."""
    dataset["TOTAL_FUEL_USED"] = (
        dataset["FUEL_USED_1"]
        + dataset["FUEL_USED_2"]
        + dataset["FUEL_USED_3"]
        + dataset["FUEL_USED_4"]
    )
    dataset["TOTAL_FOB_BY_QTY"] = (
        dataset["VALUE_FUEL_QTY_CT"]
        + dataset["VALUE_FUEL_QTY_FT1"]
        + dataset["VALUE_FUEL_QTY_FT2"]
        + dataset["VALUE_FUEL_QTY_FT3"]
        + dataset["VALUE_FUEL_QTY_FT4"]
        + dataset["VALUE_FUEL_QTY_LXT"]
        + dataset["VALUE_FUEL_QTY_RXT"]
    )
    return dataset


def drop_features(dataset):
    """Drops individual engine and tank columns after aggregation."""
    return dataset.drop(
        [
            "FUEL_USED_1",
            "FUEL_USED_2",
            "FUEL_USED_3",
            "FUEL_USED_4",
            "VALUE_FUEL_QTY_CT",
            "VALUE_FUEL_QTY_FT1",
            "VALUE_FUEL_QTY_FT2",
            "VALUE_FUEL_QTY_FT3",
            "VALUE_FUEL_QTY_FT4",
            "VALUE_FUEL_QTY_LXT",
            "VALUE_FUEL_QTY_RXT",
        ],
        axis=1,
    )


def autoencoder_dataset(dataset, min_val=None, max_val=None):
    """
    Prepares dataset for autoencoder: resamples to 5s, normalizes, splits labels/data.
    Pass min_val/max_val from training set when processing test data to prevent leakage.
    Returns (dataframe, normalized_data, labels, min_val, max_val).
    """
    dataset_tmp = dataset.copy()

    dataset_tmp["UTC_TIME"] = pd.to_datetime(dataset_tmp["UTC_TIME"])
    dataset_tmp = (
        dataset_tmp.set_index("UTC_TIME").resample("5s").mean().reset_index().dropna()
    )

    dataset_tmp = add_features(dataset_tmp)
    dataset_tmp = drop_features(dataset_tmp)

    cols_to_drop = [
        c for c in ["UTC_TIME", "FW_GEO_ALTITUDE", "FLIGHT_PHASE_COUNT", "Flight"]
        if c in dataset_tmp.columns
    ]
    dataset_tmp = dataset_tmp.drop(cols_to_drop, axis=1)

    label_col = dataset_tmp.pop("label")
    dataset_tmp["LABEL"] = label_col

    raw_data = dataset_tmp.values
    labels = raw_data[:, -1]
    data = raw_data[:, 0:-1]

    if min_val is None:
        min_val = tf.reduce_min(data)
        max_val = tf.reduce_max(data)

    data = (data - min_val) / (max_val - min_val)
    data = tf.cast(data, tf.float64)

    return dataset_tmp, data, labels, min_val, max_val


def compute_threshold(model, train_data):
    """Calculates anomaly threshold from training reconstruction error (mean + 1 std)."""
    reconstructions = model.predict(train_data)
    train_loss = tf.keras.losses.mae(reconstructions, train_data)
    return float(np.mean(train_loss) + np.std(train_loss))


def predict_anomalies(model, dataset, threshold):
    """Returns True for normal samples (reconstruction error < threshold), False for anomalies."""
    reconstructions = model.predict(dataset)
    loss = tf.keras.losses.mae(reconstructions, dataset)
    return pd.DataFrame(tf.math.less(loss, threshold))


if __name__ == "__main__":
    random.seed(42)

    # 1. Load
    model = tf.keras.models.load_model("model/my_model")
    dataset = pd.read_csv("data/msn_14_fuel_leak_signals_preprocessed.csv", sep=";")

    # 2. Flight segmentation & leak injection (before split)
    dataset = introduce_flight_column(dataset)
    dataset = create_leak(dataset, leak_flow=5.0)
    print("Label distribution:", dataset["label"].value_counts().to_dict())

    # 3. Train/test split at flight level (no row-level leakage)
    train_df, test_df = split_by_flight(dataset, test_size=0.2, random_state=42)
    print(f"Train flights: {train_df['FLIGHT'].nunique()}, Test flights: {test_df['FLIGHT'].nunique()}")

    # 4. Feature engineering applied independently to train and test
    train_df = define_new_features(train_df)
    test_df = define_new_features(test_df)

    # 5. Outlier capping: bounds fitted on train only, applied to both
    bounds = fit_outlier_bounds(train_df)
    train_df = apply_outlier_capping(train_df, bounds)
    test_df = apply_outlier_capping(test_df, bounds)

    # 6. Autoencoder dataset: normalization fitted on train, reused for test
    train_dataset, train_data, train_labels, min_val, max_val = autoencoder_dataset(train_df)
    test_dataset, test_data, test_labels, _, _ = autoencoder_dataset(test_df, min_val, max_val)

    # 7. Threshold computed from normal training samples only (no leakage)
    normal_train_data = train_data[train_labels == 0]
    threshold = compute_threshold(model, normal_train_data)
    print(f"Anomaly threshold (from train): {threshold:.6f}")

    # 8. Predictions on held-out test data
    predictions = predict_anomalies(model, test_data, threshold)

    # 9. Evaluation with full metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    y_true = (test_labels == 1).astype(int)   # 1 = leak
    y_pred = (~predictions[0]).astype(int)     # False = anomaly → predicted leak

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Leak"]))
    print(f"\nROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")

    # 10. Save results
    result = pd.concat(
        [test_dataset.reset_index(drop=True), predictions.rename(columns={0: "pred"})],
        axis=1,
    ).dropna()
    result.to_csv("test.csv", sep=";")
    print(f"\nSaved {len(result)} rows to test.csv")
