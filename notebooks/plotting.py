import re
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


class Plot:
    def plot_multi_ts(df: pd.DataFrame):
        """Plot multi time series in the same graph
        df: a pd.DataFrame containing multi time series with size (n,L) where n = number of time series"""

        df1 = df.T.reset_index(drop=True)

        fig = px.line(df1, x=df1.index, y=df1.columns)
        fig.show()
        return

    def arima_plot(y: pd.DataFrame, lags=None, figsize=(10, 5)):
        """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - a pd.DataFrame of a single time series with size (?,1)
        lags - how many lags to include in ACF, PACF calculation
        """

        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title(
            "Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}".format(p_value)
        )
        plot_acf(y, lags=lags, ax=acf_ax)
        plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        return

    def plot_forecast(train: pd.DataFrame, valid: pd.DataFrame, pred: pd.DataFrame):
        """Plot the forecasting and return the valid mae and maoe
        train: pd.DataFrame with size (?,1)
        valid: pd.DataFrame with size (?,1)
        test: pd.DataFrame with size (?,1)
        """
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_absolute_percentage_error

        valid_mae = mean_absolute_error(valid, pred)
        valid_mape = mean_absolute_percentage_error(valid, pred)

        plt.figure(figsize=(10, 5))
        plt.title(f"valid_mae: {valid_mae:.2f}, valid_mape: {valid_mape:.3f}")
        plt.plot(train, label="train")
        plt.plot(valid, label="valid")
        pred.index = valid.index
        plt.plot(pred, label="pred")
        plt.legend()
        plt.show()

        return valid_mae, valid_mape


class ModelEvaluation:
    def _extract_feature_importance(self, model):
        print("extracting feature importance (gain/split)....")

        sorted_feature_gain = sorted(
            zip(model.feature_importance(importance_type="gain"), model.feature_name()),
            reverse=True,
        )
        feature_imp_gain = pd.DataFrame(
            sorted_feature_gain, columns=["value", "feature"]
        )

        sorted_feature_split = sorted(
            zip(
                model.feature_importance(importance_type="split"), model.feature_name()
            ),
            reverse=True,
        )
        feature_imp_split = pd.DataFrame(
            sorted_feature_split, columns=["value", "feature"]
        )

        return feature_imp_gain, feature_imp_split

    @classmethod
    def plot_lgbm_feature_importance(cls, trained_model, i=1):

        feature_imp_gain, feature_imp_split = cls()._extract_feature_importance(
            trained_model
        )

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Bar(
                x=feature_imp_gain.feature.to_list(),
                y=feature_imp_gain.value.to_list(),
                name="Gain",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=feature_imp_split.feature.to_list(),
                y=feature_imp_split.value.to_list(),
                name="Split",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"LGBM Feature Importance - model{i}", width=800, height=600
        )
        fig.show()
