import pandas as pd
import random
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go


def introduce_flight_column(dataset, flight_number=0):
    # copy dataset
    dataset_tmp = dataset.copy()

    # Drop NaN values
    dataset_tmp = dataset_tmp.dropna()

    # assign flight number to rows that have indexes following each other
    # if the difference between two indexes is greater than 1, a new flight is assumed
    # print(dataset_tmp.shape)

    for i in range(1, dataset_tmp.shape[0]):
        if dataset_tmp.index[i] - dataset_tmp.index[i - 1] > 1:
            flight_number += 1
        dataset_tmp.at[dataset_tmp.index[i], "FLIGHT"] = flight_number

    # Fill nan values in flight column with 0
    dataset_tmp["FLIGHT"] = dataset_tmp["FLIGHT"].fillna(0)

    # only keep rows with flight_phase_count 8
    dataset_tmp = dataset_tmp[dataset_tmp["FLIGHT_PHASE_COUNT"] == 8]

    # for each dataset in datasets_unique_flights count occurences of FLIGHT
    counts = dataset_tmp["FLIGHT"].value_counts()
    longest_flight = (
        pd.DataFrame(counts).sort_values(by="FLIGHT", ascending=False).iloc[0].name
    )

    # only keep values that have FLIGHT == longest_flight
    dataset_tmp = dataset_tmp[dataset_tmp["FLIGHT"] == longest_flight]

    # return dataset with flight column
    return dataset_tmp


def create_leak(dataset, leak_flow, start_index=None, end_index=None):
    # create copy of dataset
    dataset_tmp = dataset.copy()

    # define FUEL_TANK_COLS
    FUEL_TANK_COLS = [
        "VALUE_FUEL_QTY_CT",
        "VALUE_FUEL_QTY_RXT",
        "VALUE_FUEL_QTY_LXT",
        "VALUE_FUEL_QTY_FT1",
        "VALUE_FUEL_QTY_FT2",
        "VALUE_FUEL_QTY_FT3",
        "VALUE_FUEL_QTY_FT4",
    ]

    # choose random fuel tank where the fuel leak is happening
    while True:
        fuel_tank = random.choice(FUEL_TANK_COLS)

        # check if fuel tank is empty at any point in time
        if dataset_tmp[fuel_tank].max() >= leak_flow:
            break

    increment = 0

    # new column label with value 0
    dataset_tmp["label"] = False

    print(start_index)

    for index in dataset_tmp[1300:].index:
        dataset_tmp.at[index, "VALUE_FOB"] -= leak_flow + increment
        dataset_tmp.at[index, fuel_tank] -= leak_flow + increment
        dataset_tmp.at[index, "label"] = True
        increment += leak_flow

    # set negative values in value_fob to 0
    dataset_tmp["VALUE_FOB"] = dataset_tmp["VALUE_FOB"].clip(lower=0)

    # set negative values in fuel_tank to 0
    dataset_tmp[fuel_tank] = dataset_tmp[fuel_tank].clip(lower=0)

    return dataset_tmp


# helper function to subtract first value of a series from all values
def subtract_first(x):
    return x.iloc[0] - x


def define_new_features(dataset):
    # copy dataset
    dataset_tmp = dataset.copy()

    # initialize TOTAL_FUEK_USED column
    dataset_tmp["TOTAL_FUEL_USED"] = 0

    # Total fuel column is the sum of all fuel used columns
    for i in range(1, 5):
        dataset_tmp["TOTAL_FUEL_USED"] += dataset_tmp["FUEL_USED_" + str(i)]

    # Get difference for VALUE_FOB for each flight
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
    ].transform(subtract_first)

    dataset_tmp["VALUE_FOB_MISSING_BY_QTY"] = dataset_tmp.groupby("FLIGHT")[
        "TOTAL_FOB_BY_QTY"
    ].transform(subtract_first)

    # new columnn VALUE_FOB_BY_FUEL_USED.
    # For ex: FOB_calculated(t=3) = VALUE_FOB(t=0) - delta_total_fuel_used(t=1) - delta_total_fuel_used(t=2) - delta_total_fuel_used(t=3)
    value_fob_0 = dataset_tmp["VALUE_FOB"].iloc[0]
    # column VALUE_FOB_BY_FUEL_USED is value_fob_0 - TOTAL_FUEL_USED
    dataset_tmp["VALUE_FOB_BY_FUEL_USED"] = value_fob_0 - dataset_tmp["TOTAL_FUEL_USED"]

    # code to remove first part of flight where the plane is on the ground and the fuel used is resetted
    for flight_num in dataset_tmp["FLIGHT"].unique():
        # get first index of flight
        first_index = dataset_tmp[dataset_tmp["FLIGHT"] == flight_num].index[0]

        # get last index of flight
        last_index = dataset_tmp[dataset_tmp["FLIGHT"] == flight_num].index[-1]

        # get location of min value of VALUE_FOB per flight
        min_value_sum_fuel_used = dataset_tmp[dataset_tmp["FLIGHT"] == flight_num][
            "TOTAL_FUEL_USED"
        ].idxmin()

        # delete in dataset_tmp all rows between first_location and min_value_fob
        dataset_tmp.drop(
            dataset_tmp.loc[first_index:min_value_sum_fuel_used].index, inplace=True
        )

    return dataset_tmp


# count number of outliers outside of 95% confidence interval
def drop_outliers(dataset):
    # create copy of dataset
    dataset_tmp = dataset.copy()

    for column in dataset_tmp.columns:
        # skip columns that are not numeric
        if dataset_tmp[column].dtype != "float64":
            continue

        # calculate 95% confidence interval
        lower_bound, upper_bound = np.percentile(dataset_tmp[column], [2.5, 97.5])

        # count number of outliers
        outliers = dataset_tmp[column][
            (dataset_tmp[column] < lower_bound) | (dataset_tmp[column] > upper_bound)
        ]

        # delete outliers
        dataset_tmp.drop(outliers.index, inplace=True)

    return dataset_tmp


def add_features(dataset):
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
    dataset = dataset.drop(
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
        1,
    )
    return dataset


def autoencoder_dataset(dataset):
    # create copy of dataset
    dataset_tmp = dataset.copy()

    # convert UTC_TIME to datetime
    dataset_tmp["UTC_TIME"] = pd.to_datetime(dataset_tmp["UTC_TIME"])

    # resample each dataset to 5 seconds
    dataset_tmp = (
        dataset_tmp.set_index("UTC_TIME").resample("5S").mean().reset_index().dropna()
    )

    dataset_tmp = add_features(dataset_tmp)

    dataset_tmp = drop_features(dataset_tmp)

    move_column1 = dataset_tmp.pop("label")

    dataset_tmp.insert(10, "LABEL", move_column1)

    dataset_tmp_with_UTC = dataset_tmp.copy()

    dataset_tmp = dataset_tmp.drop(
        ["UTC_TIME", "FW_GEO_ALTITUDE", "FLIGHT_PHASE_COUNT"], axis=1
    )

    # dataset_tmp["LABEL"].replace([1, 0], [0, 1], inplace=True)

    raw_data = dataset_tmp.values

    # The last element contains the labels
    labels = raw_data[:, -1]

    # The other data points are the data
    data = raw_data[:, 0:-1]

    # Normalize Data

    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    data = (data - min_val) / (max_val - min_val)

    data = tf.cast(data, tf.float64)

    return dataset_tmp, dataset_tmp_with_UTC, data, labels


def predict_values(model, dataset):
    reconstructions = model.predict(dataset)
    loss = tf.keras.losses.mae(reconstructions, dataset)

    # Calculate train loss
    train_loss = tf.keras.losses.mae(reconstructions, dataset)

    # Get predictions
    threshold = np.mean(train_loss) + np.std(train_loss)
    return pd.DataFrame(tf.math.less(loss, threshold))


def clean_dataset(dataset):
    dataset = introduce_flight_column(dataset)
    dataset = create_leak(dataset, 5.0, start_index=dataset.index[len(dataset) // 2])
    print(dataset["label"].value_counts())
    dataset = define_new_features(dataset)
    dataset = drop_outliers(dataset)
    return dataset


# main
if __name__ == "__main__":
    # load model
    model = tf.keras.models.load_model("model/my_model")
    dataset = pd.read_csv("data/msn_14_fuel_leak_signals_preprocessed.csv", sep=";")

    # data cleaning
    dataset = clean_dataset(dataset)

    # get dataset for autoencoder
    dataset, dataset_tmp_with_UTC, dataset_autoencoder, labels = autoencoder_dataset(
        dataset
    )

    # predict labels
    predict_values = predict_values(model, dataset_autoencoder)

    pd.concat([dataset, pd.DataFrame(predict_values)], axis=1).dropna().to_csv(
        "test.csv", sep=";"
    )

    print(dataset.shape)
    print(dataset_autoencoder.shape)
    print(pd.DataFrame(predict_values).shape)
