import re
import numpy as np
import pandas as pd


class TsHandler:
    def create_single_time_series(df: pd.DataFrame, name: str):
        """Create a single time series with size (L,1)
        df: a pd.DataFrame containing multi time series with size (n,L) where n = number of time series"""

        df_s = df[df.index == name].reset_index(drop=True).T.reset_index(drop=True)
        df_s.columns = ["sales"]
        df_s.name = name
        df_s
        return df_s

    def create_diff(df: pd.DataFrame):
        """diff the df of single time series"""
        df1 = pd.DataFrame(df["sales"].diff(1).dropna().reset_index(drop=True))
        df1.name = df.name
        return df1

    def tabularise_single_ts(data: np.array, window_size=6, step=1):
        """Tabularise a single time series into a table df
        data: a np.array of single time series with size (?,)
        window_size: w
        step: next i step prediction [1, HORIZON]
        """
        X, y = [], []

        for i in range(len(data) - (window_size + step) + 1):
            window = data[i : i + window_size]
            target = data[i + window_size + step - 1]
            X.append(window)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        window_cols = [f"x{i+1}" for i in range(window_size)]
        df_X = pd.DataFrame(X, columns=window_cols)

        target_cols = [f"y{step}"]
        df_y = pd.DataFrame(y, columns=target_cols)

        df = pd.concat([df_X, df_y], axis=1)
        df.name = f"step{step}"

        return df
