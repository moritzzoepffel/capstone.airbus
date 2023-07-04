import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import airbusclean
import tensorflow as tf

# import keras


#### COLORS TO USE: https://www.designpieces.com/palette/airbus-color-palette-hex-and-rgb/


def make_flight_diff_dataset(df):
    df_copy = df.copy()  # Create a copy of the DataFrame
    colstodiff = [col for col in df.columns if col not in ["UTC_TIME", "MSN", "Flight"]]
    df_copy[colstodiff] = df_copy[colstodiff].diff()
    df_copy = df_copy.dropna()
    return df_copy


def main():
    #### PAGE CONFIGURATION ####
    st.set_page_config(
        page_title="Airbus Capstone",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    #### SIDEBAR ####
    st.sidebar.image("./airbusdash/Airbus-Logo.png", width=300)

    ### Upload dataset ####
    st.sidebar.subheader(
        "Upload Dataset"
    )  # Until we dont upload a dataset nothing happens
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is None:  # When no file is uploaded
        st.title("Maintenance Dashboard")
        st.write(
            """The following system has been developed to support Airbus maintenance engineers in their daily work.
                By deploying Atificial Intelligence and Machine Learning algorithms, 
                the system is able to preliminalry detect fuel leaks and another anomalies in the fuel system of the aircraft."""
        )

        st.write(
            """Use the side bar to navigate through the different pages. Upload a dataset to get started with the analysis.
                By clicking on the buttons data is transformed and visualized accordingly"""
        )
        with st.expander("Dataset Requirements"):
            st.write("Here put a description of the data cleaning applied. ")

    elif uploaded_file is not None:  # When someone uploads
        #### PROCESS THE DATASETS ####
        df = pd.read_csv(uploaded_file)  # turn the file to a df (TRY EXCEPT ADDITION)
        df.UTC_TIME = pd.to_datetime(df.UTC_TIME)
        df.FLIGHT_PHASE_COUNT = df.FLIGHT_PHASE_COUNT.astype(int, errors="ignore")  #
        #### AIRBUS CLEAN SCRIPT ####
        processed_df = airbusclean.clean_dataset(df)

        # Sidebar options
        st.sidebar.subheader("Select a dashboard:")
        sidebar_options = ["Flight Analytics", "Cruise Analysis", "Leakage Detection"]
        selected_option = st.sidebar.radio("Select an option", sidebar_options)

        if selected_option == "Flight Analytics":
            st.title("Flight Analytics")
            st.write("Provide an overview of the dataset and any relevant information.")

            st.write(f"Selected flight: {df.Flight.unique()[0]}")
            st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

            phase_dict = {
                1: "Pre-flight",
                2: "Engine Run",
                3: "Take-Off 1",
                4: "Take-Off 2",
                5: "Take-Off 3",
                6: "Climbing 1",
                7: "Climbing 2",
                8: "Cruise",
                9: "Descent",
                10: "Approach",
                11: "Landing",
                12: "Post-flight",
            }

            ### Break Line ###
            st.markdown("<hr>", unsafe_allow_html=True)

            ## MEtric cards ##
            col1, col2, col3 = st.columns(3)
            fobconsumed = df.VALUE_FOB.max() - df.VALUE_FOB.min()
            col1.metric("Value FOB Consumed", f"{fobconsumed:.2f}")

            altitude_max = df.FW_GEO_ALTITUDE.max()
            col2.metric("Max Altitude", f"{altitude_max:.2f}")

            Flightduration = df.UTC_TIME.max() - df.UTC_TIME.min()
            col3.metric("Flight Duration", f"{Flightduration}")
            # Apply CSS
            col1.markdown(
                "<style>div[data-testid='stMetric-value']{ text-align: center; }</style>",
                unsafe_allow_html=True,
            )
            col2.markdown(
                "<style>div[data-testid='stMetric-value']{ text-align: center; }</style>",
                unsafe_allow_html=True,
            )
            col3.markdown(
                "<style>div[data-testid='stMetric-value']{ text-align: center; }</style>",
                unsafe_allow_html=True,
            )

            ### TOP GRAPH ###
            fig = (
                go.Figure(
                    data=go.Scatter(
                        x=df["UTC_TIME"],
                        y=df["FW_GEO_ALTITUDE"],
                        mode="lines",
                        name="Altitude",
                    )
                )
                .add_trace(
                    go.Scatter(
                        x=df["UTC_TIME"],
                        y=df["VALUE_FOB"],
                        mode="lines",
                        name="Value FOB",
                    )
                )
                .update_layout(
                    title="Altitude and Value Fuel on Board (FOB)",
                    width=1000,
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
            )

            # Render the plot in Streamlit
            st.plotly_chart(fig)

            left, right = st.columns(2)
            with left:
                phaselabels = [
                    phase_dict.get(phase, "Unknown")
                    for phase in df["FLIGHT_PHASE_COUNT"]
                ]

                fig5 = go.Figure(
                    go.Pie(
                        labels=phaselabels,
                        values=df.FLIGHT_PHASE_COUNT,
                        name="Flight Phases",
                    )
                ).update_layout(
                    title="Flight Phase Distribution",
                    width=600,
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig5)

                fig = (
                    go.Figure(
                        data=go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["FUEL_USED_1"],
                            mode="lines",
                            name="1",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["FUEL_USED_2"],
                            mode="lines",
                            name="2",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["FUEL_USED_3"],
                            mode="lines",
                            name="3",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["FUEL_USED_4"],
                            mode="lines",
                            name="4",
                        )
                    )
                    .update_layout(
                        title=f"Fuel Consumption (Engine 1 to 4)",
                        xaxis_title="Time",
                        yaxis_title=f"Fuel Consumption (L)",
                        legend_title="Engines",
                        width=600,
                        height=400,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )
                )

                st.plotly_chart(fig)

                phasedurationdf = df.groupby("FLIGHT_PHASE_COUNT")["UTC_TIME"].agg(
                    ["min", "max"]
                )
                phasedurationdf.index = phasedurationdf.index.map(phase_dict)
                phasedurationdf = phasedurationdf.rename(
                    columns={"min": "Start Time", "max": "End Time"}
                )
                st.write(phasedurationdf)

            with right:
                fig6 = (
                    go.Figure(
                        data=go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FOB"],
                            mode="lines",
                            name="Value FOB",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FUEL_QTY_LXT"],
                            mode="lines",
                            name="Left Tank",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FUEL_QTY_RXT"],
                            mode="lines",
                            name="Right Tank",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FUEL_QTY_CT"],
                            mode="lines",
                            name="Central Tank",
                        )
                    )
                    .update_layout(
                        title=f"Fuel Tank Quantities (1-4)",
                        xaxis_title="Time",
                        yaxis_title=f"Fuel Tank Quantity (Engine 1-4)",
                        legend_title="Fuel Tank: ",
                        width=600,
                        height=400,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )
                )
                st.plotly_chart(fig6)

                fig2 = (
                    go.Figure(
                        data=go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FUEL_QTY_FT1"],
                            mode="lines",
                            name="1",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FUEL_QTY_FT2"],
                            mode="lines",
                            name="2",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FUEL_QTY_FT3"],
                            mode="lines",
                            name="3",
                        )
                    )
                    .add_trace(
                        go.Scatter(
                            x=df["UTC_TIME"],
                            y=df["VALUE_FUEL_QTY_FT4"],
                            mode="lines",
                            name="4",
                        )
                    )
                    .update_layout(
                        title=f"Fuel Tank Quantities (1-4)",
                        xaxis_title="Time",
                        yaxis_title=f"Fuel Tank Quantity (Engine 1-4)",
                        legend_title="Fuel Tank: ",
                        width=600,
                        height=400,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )
                )

                st.plotly_chart(fig2)

            ### Select a column to plot ###
            st.subheader("Select a column:")
            cols_for_selection = [
                col
                for col in df.columns
                if col not in ["UTC_TIME", "MSN", "Flight", "FLIGHT", "DATE", "TIME"]
            ]
            selected_column = st.selectbox("Select a column", cols_for_selection)

            # fig for selectbox
            fig1 = go.Figure(
                data=go.Scatter(
                    x=df["UTC_TIME"], y=df[f"{selected_column}"], mode="lines"
                )
            ).update_layout(
                title=f"{selected_column} over Time",
                xaxis_title="Time",
                yaxis_title=f"{selected_column}",
                width=1000,
                height=500,
            )
            st.plotly_chart(fig1)

        #### PROCESS DATASET ####
        elif selected_option == "Cruise Analysis":
            diff_processed_df = make_flight_diff_dataset(processed_df)
            # two dfs at this point: processed_df and diff_processed_df

            st.title("Cruise Phase Flight Analysis")
            with st.expander("Data cleaning methodology"):
                st.write("Here put a description of the data cleaning applied. ")

            st.subheader("Engineered Features")
            st.write("Provide an overview of the dataset and any relevant information.")

            st.write(
                f"Rows: {processed_df.shape[0]} | Columns: {processed_df.shape[1]}"
            )

            st.write(f"{processed_df.columns}")

            ### Break Line ###
            st.markdown("<hr>", unsafe_allow_html=True)

            ## MEtric cards ##
            col1, col2, col3 = st.columns(3)
            flight_value = processed_df.Flight.unique()[0]
            col1.metric("Flight", f"{flight_value}")

            altitude_rate = diff_processed_df.FW_GEO_ALTITUDE.mean()
            col2.metric("Mean Altitude rate of change", f"{altitude_rate:.2f}")

            fob_rate = diff_processed_df.VALUE_FOB.mean()
            col3.metric("Mean Fob rate of change", f"{fob_rate:.2f}")
            # Apply CSS
            col1.markdown(
                "<style>div[data-testid='stMetric-value']{ text-align: center; }</style>",
                unsafe_allow_html=True,
            )
            col2.markdown(
                "<style>div[data-testid='stMetric-value']{ text-align: center; }</style>",
                unsafe_allow_html=True,
            )
            col3.markdown(
                "<style>div[data-testid='stMetric-value']{ text-align: center; }</style>",
                unsafe_allow_html=True,
            )

            ## Plotting ##
            col1, col2 = st.columns(2)

            with col1:
                fig6 = go.Figure()

                fig6.add_trace(
                    go.Scatter(
                        x=processed_df.UTC_TIME,
                        y=processed_df.VALUE_FOB,
                        mode="lines",
                        name="Value FOB",
                    )
                )
                fig6.add_trace(
                    go.Scatter(
                        x=processed_df.UTC_TIME,
                        y=processed_df.TOTAL_FOB_BY_QTY,
                        mode="lines",
                        name="Total FOB by Tank QTY",
                    )
                )
                fig6.update_layout(
                    title=f"Value FOB vs Total FOB by Tank QTY",
                    width=600,
                    height=400,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                st.plotly_chart(fig6)

                fig8 = go.Figure()

                fig8.add_trace(
                    go.Scatter(
                        x=processed_df.UTC_TIME,
                        y=processed_df.VALUE_FOB_MISSING,
                        mode="lines",
                        name="Value FOB Consumption",
                    )
                )
                fig8.add_trace(
                    go.Scatter(
                        x=processed_df.UTC_TIME,
                        y=processed_df.TOTAL_FUEL_USED,
                        mode="lines",
                        name="Total Fuel Used by Engines",
                    )
                )
                fig8.update_layout(
                    title=f"Value FOB Consumption vs Total Fuel Used by Engines",
                    width=600,
                    height=400,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                st.plotly_chart(fig8)

            with col2:
                fig7 = go.Figure()

                fig7.add_trace(
                    go.Scatter(
                        x=processed_df.UTC_TIME,
                        y=processed_df.DELTA_VFOB_VS_VFOBQTY,
                        mode="lines",
                    )
                )
                fig7.update_layout(
                    title="Delta Value FOB and FOB by Tank Quantities",
                    width=600,
                    height=400,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                st.plotly_chart(fig7)

                fig9 = go.Figure()

                fig9.add_trace(
                    go.Scatter(
                        x=processed_df.UTC_TIME,
                        y=processed_df.ALTITUDE_DIFF,
                        mode="lines",
                    )
                )
                fig9.update_layout(
                    title="Altitude Difference",
                    width=600,
                    height=400,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                st.plotly_chart(fig9)

            st.subheader("Derivative Data")

            st.write(diff_processed_df.head(5))

            st.write(f"{diff_processed_df.columns}")

        elif selected_option == "Leakage Detection":
            st.header("Leakage Detection")

            # tf.compat.v1.disable_eager_execution()
            # load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

            # loaded_model = tf.saved_model.load(model_path)
            # loaded_model = tf.saved_model.load(model_path, options=load_options)

            # loaded_model = tf.keras.models.load_model(model_path)
            # loaded_model = tf.keras.models.load_model(model_path, compile=False, options=load_options)

            # loaded_model = tf.keras.saving.load_model(model_path, compile=False, safe_mode=True)

            # Define the costs
            cost_true_negative = 0
            cost_true_positive = 0
            cost_false_negative = 15  # Cost of a false negative
            cost_false_positive = 10  # Cost of a false positive

            def modified_cost(y_true, y_pred):
                mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
                fn = tf.keras.backend.sum(
                    tf.keras.backend.square(y_true - y_pred)
                    * tf.keras.backend.cast(
                        tf.keras.backend.equal(y_true, 1), tf.keras.backend.floatx()
                    )
                )
                fp = tf.keras.backend.sum(
                    tf.keras.backend.square(y_true - y_pred)
                    * tf.keras.backend.cast(
                        tf.keras.backend.equal(y_true, 0), tf.keras.backend.floatx()
                    )
                )
                tn = tf.keras.backend.sum(
                    tf.keras.backend.square(y_true - y_pred)
                    * tf.keras.backend.cast(
                        tf.keras.backend.equal(y_true, 0), tf.keras.backend.floatx()
                    )
                )
                tp = tf.keras.backend.sum(
                    tf.keras.backend.square(y_true - y_pred)
                    * tf.keras.backend.cast(
                        tf.keras.backend.equal(y_true, 1), tf.keras.backend.floatx()
                    )
                )
                return (
                    mse
                    + cost_false_negative * fn
                    + cost_false_positive * fp
                    + cost_true_negative * tn
                    + cost_true_positive * tp
                )

            with tf.keras.utils.custom_object_scope({"modified_cost": modified_cost}):
                new_model = tf.keras.models.load_model(
                    "./model/my_model", compile=False
                )

            # Check its architecture
            (
                dataset,
                dataset_tmp_with_UTC,
                dataset_autoencoder,
                labels,
            ) = airbusclean.autoencoder_dataset(processed_df)

            preds = airbusclean.predict_values(new_model, dataset_autoencoder)

            concatted_dataset = pd.concat(
                [dataset_tmp_with_UTC, preds], axis=1
            ).dropna()

            # rename column named 0 to PREDICTION
            concatted_dataset.rename(columns={0: "PREDICTION"}, inplace=True)

            # replace 0 with FALSE and 1 with TRUE in column LABEL
            concatted_dataset["LABEL"] = concatted_dataset["LABEL"].replace(
                [0, 1], [False, True]
            )

            pred_fig = go.Figure()

            pred_fig.add_trace(
                go.Scatter(
                    x=concatted_dataset.UTC_TIME,
                    y=concatted_dataset.VALUE_FOB,
                    mode="lines",
                    name="VALUE_FOB",
                )
            )

            # get UTC_TIME of first True value in LABEL column. UTC TIME is not index
            first_true_index = concatted_dataset[
                concatted_dataset["LABEL"] == True
            ].iloc[0]["UTC_TIME"]

            # from first true index to end of dataset color background red
            pred_fig.add_vrect(
                x0=first_true_index,
                x1=concatted_dataset.iloc[-1]["UTC_TIME"],
                fillcolor="red",
                opacity=0.25,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(pred_fig)

            st.write(concatted_dataset)


if __name__ == "__main__":
    main()
