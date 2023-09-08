import numpy as np
import pandas as pd

import lightgbm as lgbm

from plotting import ModelEvaluation
from time_series import TsHandler
from preprocess import Preprocessor, Dataset


class Prediction:
    def recurrent_predict(model, trainset_last_row, horizon):
        valid_y_pred = []
        input = trainset_last_row

        for i in np.arange(horizon):
            y_pred = model.predict(input.reshape(1, -1))
            input = np.append(input[1:], y_pred)
            valid_y_pred.append(y_pred)

        return np.array(valid_y_pred)

    def direct_multi_output_approach(
        df_s: pd.DataFrame,
        col: str,
        windowsize: int,
        horizon: int,
        lgbm_params: dict,
        lgbm_trainer_args: dict,
        show_plot=False,
    ):

        train = df_s.iloc[:-horizon]
        input = train.iloc[-windowsize:].T
        list_lgbm_model = []
        y_pred_concat = []

        for i in np.arange(1, horizon + 1):
            print(f"---------------------Step{i}---------------------")
            df = TsHandler.tabularise_single_ts(
                df_s[f"{col}"].values, window_size=windowsize, step=i
            )
            df = Preprocessor.set_train_valid_test(df, horizon=horizon)

            num_cols = [f"x{i+1}" for i in range(windowsize)]
            target_col = f"y{i}"
            dataset_all = Dataset(
                df, cat_cols=[], num_cols=num_cols, target_col=target_col
            )

            lgbm_model = lgbm.train(
                lgbm_params,
                **lgbm_trainer_args,
                train_set=dataset_all.create_lgbm_dataset("TRAIN"),
                valid_sets=dataset_all.create_lgbm_dataset("VALID"),
            )
            list_lgbm_model.append(lgbm_model)
            if show_plot:
                ModelEvaluation.plot_lgbm_feature_importance(lgbm_model, i)

            y_pred_single_ts = lgbm_model.predict(input)
            y_pred_concat.append(y_pred_single_ts)

        return np.array(y_pred_concat), list_lgbm_model
