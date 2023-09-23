import re
import numpy as np
import pandas as pd

from preprocess import Preprocessor


class TsHandler:
    # TODO: classmethod?
    def create_single_time_series(df: pd.DataFrame, name: str):
        """
        Create a single time series with size (L,1) by selecting the name

        df: a pd.DataFrame with item_id as index and all timesteps are the columns
        name: str of item_id
        """

        df_s = df[df.index == name].reset_index(drop=True).T.reset_index(drop=True)
        df_s.columns = ["sales"]
        df_s.name = name
        df_s
        return df_s

    def create_training_data(
        df_item_id: pd.DataFrame, names: list, windowsize: int, horizon: int, step: int
    ):
        """
        Create a tabularised training data with
        x1 to x_windowsize as features
        and y_step as target

        df_item_id: a pd.DataFrame with item_id as index and all timesteps are the columns
        names: a list of selected item_id (str)
        windowsize: windowsize
        horizon: horizon
        step: the step of y_step or the steo of the model

        return:
        df_m: a pd.DataFrame
        """
        df_m = pd.DataFrame()

        for name in names:
            df_s = TsHandler.create_single_time_series(df_item_id, name=name)
            df = TsHandler.tabularise_single_ts(
                df_s["sales"].values, window_size=windowsize, step=step
            )
            df = Preprocessor.set_train_valid_test(df, horizon=horizon)
            df["item_id"] = name
            df_m = df_m.append(df)

        return df_m

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
