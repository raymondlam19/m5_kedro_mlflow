import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
import warnings

warnings.filterwarnings("ignore")


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
