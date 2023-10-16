import numpy as np
import pandas as pd

import lightgbm as lgbm

from plotting import Plot, ModelEvaluation
from time_series import TsHandler
from preprocess import Preprocessor, Dataset


class Prediction:
    def recurrent_predict(trained_model, trainset_last_row: np.array, horizon: int):
        """
        Recurrently predict using pervious prediction as input with the trained model
        which the model was trained with y1 as true label
        (prediction + concat prediction)
        i.e.
        - Prediction(t+1) = model(obs(t), obs(t-1), …, obs(t-w))
        - Prediction(t+2) = model(prediction(t+1) , obs(t), obs(t-1), …, obs(t-w+1))

        trained_model       : trained model
        trainset_last_row   : the last row of array with size (w,) in training data
        horizon             : number of forecasting timestep
        """

        valid_y_pred = []
        input = trainset_last_row

        for i in np.arange(horizon):
            y_pred = trained_model.predict(input.reshape(1, -1))
            input = np.append(input[1:], y_pred)
            valid_y_pred.append(y_pred)

        return np.array(valid_y_pred)

    def multi_model_approach(
        df_item_id: pd.DataFrame,
        names: list,
        windowsize: int,
        horizon: int,
        lgbm_params: dict,
        lgbm_trainer_args: dict,
        show_plot=False,
    ):
        """
        Multi model approach - 1 model for 1 timestep prediction
        (tabularise data + set train valid test + train + prediction + concat prediction)
        i.e.
        - Prediction(t+1) = model_1(obs(t), obs(t-1), …, obs(t-w))
        - Prediction(t+2) = model_2(obs(t), obs(t-1), …, obs(t-w))
        - ...
        - Prediction(t+n) = model_n(obs(t), obs(t-1), …, obs(t-w))

        df_item_id          : a pd.DataFrame containing all item_id lvl time series
        names               : a list of item_id name (str), len(names)=1 -> single ts
        windowsize          : window size of training data
        horizon             : number of forecasting timestep
        lgbm_params         : a dict containing lgbm params like num_bin, learning_rate
        lgbm_trainer_args   : a dict containing lgbm args like epoch, early stop
        show_plot           : Boolean to show feature importance of each trained model
        """

        # Return a df_y for validation
        df_y = (
            df_item_id[df_item_id.index.isin(names)]
            .T.iloc[-horizon:][names]
            .reset_index(drop=True)
        )

        list_lgbm_model = []
        y_pred_concat = []

        for i in np.arange(1, horizon + 1):
            print(f"---------------------Step{i}---------------------")
            df_m = TsHandler.create_training_data(
                df_item_id, names, windowsize, horizon, step=i
            )

            # Create lgbm dataset for lgbm training
            num_cols = [f"x{i+1}" for i in range(windowsize)]
            cat_cols = []
            features = num_cols + cat_cols
            target_col = f"y{i}"
            dataset_all = Dataset(
                df_m, cat_cols=cat_cols, num_cols=num_cols, target_col=target_col
            )

            # lgbm training
            lgbm_model = lgbm.train(
                lgbm_params,
                **lgbm_trainer_args,
                train_set=dataset_all.create_lgbm_dataset("TRAIN"),
                valid_sets=dataset_all.create_lgbm_dataset("VALID"),
            )

            # Save model
            list_lgbm_model.append(lgbm_model)

            # Plot feature importance
            if show_plot:
                ModelEvaluation.plot_lgbm_feature_importance(lgbm_model, i)

            # Single time step prediction
            # Select last row of features in TRAIN for each item_id
            idx = len(df_item_id.columns) - windowsize - horizon
            input = df_m.loc[df_m.index == idx, features]
            y_pred = lgbm_model.predict(input)
            y_pred_concat.append(y_pred)

        df_y_pred = pd.DataFrame(np.array(y_pred_concat), columns=names)

        return df_y, df_y_pred, list_lgbm_model
