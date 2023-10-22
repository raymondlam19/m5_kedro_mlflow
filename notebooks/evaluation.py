import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

from time_series import TsHandler
from plotting import Plot

import warnings

warnings.filterwarnings("ignore")


class Evaluation:
    def single_item_evaluate(df_item_id, name, horizon, df_y_pred):
        """
        Single item evaluation and plot the forecast.
        And return the valid mae & valid smape for the item.

        df_item_id  : a pd.DataFrame containing all item_id lvl time series
        name        : the name of the item_id
        horizon     : number of forecasting timestep
        df_y_pred   : a df of y_pred of validation period. shape: (28, len(names))
        """
        df_s = TsHandler.create_single_time_series(df_item_id, name=name)
        train = df_s.iloc[:-horizon]
        valid = df_s.iloc[-horizon:]
        y_pred = df_y_pred[[name]]
        print(
            f"\n----------------------------------------------- {name} -----------------------------------------------"
        )
        valid_mae_lgbm, valid_smape_lgbm = Plot.plot_forecast(
            train.iloc[-horizon * 4 :], valid, y_pred
        )

        return valid_mae_lgbm, valid_smape_lgbm

    def overall_evaluate(df_y, df_y_pred):
        """
        Print Overall mae & smape

        df_y        : a df of y for validation. shape: (28, len(names))
        df_y_pred   : a df of y_pred of validation period. shape: (28, len(names))
        """
        assert set(df_y.columns) == set(
            df_y_pred.columns
        ), "df_y_pred.columns doesn't match df_y.columns"
        names = df_y.columns
        print(f"Overall mae: {Metric.mae_display_str(df_y[names], df_y_pred[names])}")
        print(
            f"Overall smape: {Metric.smape_display_str(df_y[names], df_y_pred[names])}"
        )


class Metric:
    def _check_and_convert_to_np_array(y):
        """Check y and convert y into an np.array if y is not an array

        y: pd.DataFrame / pd.Series / np.array
        """
        assert (
            isinstance(y, np.ndarray)
            or isinstance(y, pd.Series)
            or isinstance(y, pd.DataFrame)
        ), "y must be a np.array, pd.Series, or pd.DataFrame"

        if isinstance(y, pd.Series):
            y = pd.DataFrame(y)

        if isinstance(y, pd.DataFrame):
            y = y.values
        else:
            y
        return y

    def smape(y_pred: np.array, y: np.array):
        """calculate smape"""
        y_pred = Metric._check_and_convert_to_np_array(y_pred)
        y = Metric._check_and_convert_to_np_array(y)
        return np.mean(np.mean(np.abs(y_pred - y) / (np.abs(y) + np.abs(y_pred))))

    def smape_display_str(y_pred: np.array, y: np.array):
        """smape string for display"""
        return f"{Metric.smape(y_pred, y)*100:0.2f}%"

    def mae_display_str(y_pred: np.array, y: np.array):
        """mae string for display"""
        return f"{mae(y_pred, y):0.3f}"


# Unit test
if __name__ == "__main__":
    df_a = pd.DataFrame({"col1": [2, 3, 4, 5], "col2": [6, 7, 8, 9]})
    df_f = pd.DataFrame({"col1": [1, 3, 5, 4], "col2": [6, 7, 10, 7]})
    print(f"mae: {mae(df_f, df_a)}")
    print(f"smape: {Metric.smape(df_f, df_a)}")

    print(f"mae: {Metric.mae_display_str(df_f, df_a)}")
    print(f"smape: {Metric.smape_display_str(df_f, df_a)}")

    np_a = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    np_f = np.array([1, 3, 5, 4, 6, 7, 10, 7])
    print(f"mae: {Metric.mae_display_str(np_f, np_a)}")
    print(f"smape: {Metric.smape_display_str(np_f, np_a)}")
