import re
import numpy as np
import pandas as pd


class TsHandler:
    def create_single_time_series(df: pd.DataFrame, name: str):
        """
        Create a single time series with size (L,1) by selecting the name

        df: a pd.DataFrame containing multi time series with size (n,L) where n = number of time series
        """

        df_s = df[df.index == name].reset_index(drop=True).T.reset_index(drop=True)
        df_s.columns = ["sales"]
        df_s.name = name
        df_s
        return df_s

    def create_diff(df_s: pd.DataFrame):
        """
        Diff the single time series

        df_s: a pd.DataFrame containing a single time series
        """

        df = pd.DataFrame(df_s["sales"].diff(1).dropna().reset_index(drop=True))
        df.name = df_s.name
        return df

    def tabularise_single_ts(data: np.array, window_size: int, step=1):
        """
        Tabularise a single time series into a table df

        data: an np.array of single time series with size (?,)
        window_size: w
        step: next i step prediction [1, HORIZON], default is 1 which means predict the next timestep
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


if __name__ == "__main__":
    # unit test
    df = TsHandler.tabularise_single_ts(np.arange(20), window_size=6, step=2)
    print(df)
