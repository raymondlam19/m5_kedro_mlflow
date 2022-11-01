"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

import logging
from typing import Any, Dict, Tuple

import os
import numpy as np
import pandas as pd

from m5_kedro_mlflow.pipelines.data_engineering.utils import create_lag_ma


def read_data(params_dataset):
    """Read 3 raw data csv -> trim and preprocess -> merge 3 df into 1
        params_dataset, params_trimming, params_features
    Args:
        parameters: Parameters defined in parameters.yml.
    Returns:
        df: trimed & merged dataframe
    """
    DIR = params_dataset["dir"]
    DF_SALES_FILENAME = params_dataset["df_sales"]
    DF_CALENDAR_FILENAME = params_dataset["df_calendar"]
    DF_PRICES_FILENAME = params_dataset["df_prices"]

    df_sales = pd.read_csv(os.path.join(DIR, DF_SALES_FILENAME))
    df_calendar = pd.read_csv(
        os.path.join(DIR, DF_CALENDAR_FILENAME), parse_dates=["date"]
    )
    df_prices = pd.read_csv(os.path.join(DIR, DF_PRICES_FILENAME))
    return df_sales, df_calendar, df_prices


def trim_and_preprocess_data(
    df_sales, df_calendar, df_prices, params_base, params_trimming
):
    TARGET_COL = params_base["target_col"]
    KEY_COLS = params_base["key_cols"]
    TEST_SIZE = params_trimming["test_size"]
    LAG = params_trimming["lag"]
    MA = params_trimming["ma"]
    START_DATE = params_trimming["start_date"]

    # Add null sales for the remaining days 1942-1969
    # d_1 to d_1941: train set & valid set
    # d_1942 - d_1969: test set (forecast F1 to F28 in sample submission)
    series_d = pd.Series(df_sales.columns)
    series_d = series_d[series_d.str.contains("d_")].reset_index(drop=True)
    series_d = series_d.apply(lambda x: x.split("_")[1]).astype(int)
    max_d = max(series_d.values)
    for d in range(max_d + 1, max_d + 1 + TEST_SIZE):
        col = "d_" + str(d)
        df_sales[col] = np.nan

    # trim
    start_date = pd.to_datetime(START_DATE) - pd.Timedelta(max(LAG) + max(MA), "days")
    df_calendar_trim = df_calendar[(df_calendar.date >= start_date)].copy()

    # For trimming
    d_min = int(df_calendar_trim.d.min().split("_")[1])
    d_max = int(df_calendar_trim.d.max().split("_")[1])
    week_min = df_calendar_trim.wm_yr_wk.min()
    week_max = df_calendar_trim.wm_yr_wk.max()

    # preprocess on calendar
    df_calendar_trim["is_holiday"] = (
        df_calendar_trim["event_name_1"].notnull()
        | df_calendar_trim["event_name_2"].notnull()
    )
    df_calendar_trim["is_weekend"] = df_calendar_trim.weekday.isin(
        ["Saturday", "Sunday"]
    )
    df_calendar_trim = df_calendar_trim[
        ["date", "wm_yr_wk", "weekday", "d", "is_holiday", "is_weekend"]
    ]

    # trim on sales
    df_sales_trim = df_sales[
        [col for col in KEY_COLS if col != "date"]
        + [f"d_{n}" for n in range(d_min, d_max + 1)]
    ]

    # trim on prices
    df_prices_trim = df_prices[
        (df_prices.wm_yr_wk >= week_min) & (df_prices.wm_yr_wk <= week_max)
    ]

    # merge 3 df into 1
    df = pd.melt(
        df_sales_trim,
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d",
        value_name=TARGET_COL,
    )
    df = pd.merge(df, df_calendar_trim, how="left", on="d")
    df = pd.merge(
        df, df_prices_trim, how="left", on=["store_id", "item_id", "wm_yr_wk"]
    )
    df["d"] = df["d"].apply(lambda x: x.split("_")[1]).astype(int)

    return df


def create_lag_ma_features_and_trim_again(df, params_trimming, params_features):
    LAG = params_trimming["lag"]
    MA = params_trimming["ma"]
    START_DATE = params_trimming["start_date"]
    MA_LAG_FEATURE = params_features["ma_lag_feature"]

    for key in MA_LAG_FEATURE:
        df = create_lag_ma(df, LAG, MA, key)

    # trimming on start & end date (due to lag & ma)
    df = df[(df.date >= START_DATE)].copy()

    # trimming on sell price
    df = df[df.sell_price.notnull()].reset_index(drop=True)
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    return df
