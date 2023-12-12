import re
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

    def multi_model_train_predict(
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

        for step in np.arange(1, horizon + 1):
            print(f"---------------------Step{step}---------------------")
            df_m = TsHandler.create_training_data(
                df_item_id, names, windowsize, horizon, step=step
            )

            # update index = index of y{step}
            df_m.index = df_m.index + windowsize + step - 1

            # Create lgbm dataset for lgbm training
            num_cols = [f"x{i+1}" for i in range(windowsize)]
            cat_cols = []
            target_col = f"y{step}"
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
                ModelEvaluation.plot_lgbm_feature_importance(lgbm_model, step)

            # Single time step prediction
            # Select the step th row of features in VALID for each item_id
            idx = min(df_m[df_m.train_valid_test=='VALID'].index) + step - 1
            features = dataset_all.num_cols + dataset_all.cat_encoded_cols
            input = df_m.loc[df_m.index == idx, features]
            y_pred = lgbm_model.predict(input)
            y_pred_concat.append(y_pred)

        df_y_pred = pd.DataFrame(np.array(y_pred_concat), columns=names)

        return df_y, df_y_pred, list_lgbm_model

    def multi_model_train_predict_exo_features(
        df_item_id: pd.DataFrame,
        df_calendar: pd.DataFrame,
        df_prices: pd.DataFrame,
        names: list,
        windowsize: int,
        horizon: int,
        lgbm_params: dict,
        lgbm_trainer_args: dict,
        exo_cat_cols: list = [],
        exo_num_cols: list = [],
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
        df_calendar         : calendar.csv
        df_prices           : sell_prices.csv then agg to item_id lvl
        names               : a list of item_id name (str), len(names)=1 -> single ts
        windowsize          : window size of training data
        horizon             : number of forecasting timestep
        lgbm_params         : a dict containing lgbm params like num_bin, learning_rate
        lgbm_trainer_args   : a dict containing lgbm args like epoch, early stop
        exo_cat_cols        : e.g. ['weekday', 'holiday'] or []
        exo_num_cols        : e.g. ['sell_price'] or []
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

        for step in np.arange(1, horizon + 1):
            print(f"---------------------Step{step}---------------------")
            df_m = TsHandler.create_training_data(
                df_item_id, names, windowsize, horizon, step=step
            )
                    
            # update index = index of y{step}
            df_m.index = df_m.index + windowsize + step - 1

            # join calendar & price
            df_m = df_m.merge(
                df_calendar[["weekday_encoded", "holiday_encoded", "wm_yr_wk"]],
                left_index=True,
                right_index=True,
                how="left",
            )

            # preserve the index for prediction
            df_m = df_m.reset_index().set_index(['item_id', 'wm_yr_wk'])

            df_m = df_m.merge(
                df_prices.set_index(['item_id', 'wm_yr_wk']),
                left_index=True,
                right_index=True,
                how="left",
            )

            # resume the index
            df_m = df_m.reset_index().set_index("index")

            # sort df_m by names
            df_m = df_m.sort_values(by='item_id', key=lambda x: x.map({name: i for i, name in enumerate(names)}))

            assert df_m.columns.isin(["weekday_encoded", "holiday_encoded", "sell_price"]).sum() == 3, "Error: join calendar & price!"

            # Create lgbm dataset for lgbm training
            num_cols = [f"x{i+1}" for i in range(windowsize)] + exo_num_cols
            target_col = f"y{step}"
            dataset_all = Dataset(
                df_m, cat_cols=exo_cat_cols, num_cols=num_cols, target_col=target_col
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
            # Select the step th row of features in VALID for each item_id
            idx = min(df_m[df_m.train_valid_test=='VALID'].index) + step - 1
            features = dataset_all.num_cols + dataset_all.cat_encoded_cols
            input = df_m.loc[df_m.index == idx, features]
            y_pred = lgbm_model.predict(input)  #shape:(len(names),)
            y_pred_concat.append(y_pred)        #len:step

        df_y_pred = pd.DataFrame(np.array(y_pred_concat), columns=names)

        return df_y, df_y_pred, list_lgbm_model

    def multi_model_predict_exo_features(
        df_item_id: pd.DataFrame,
        df_calendar: pd.DataFrame,
        df_prices: pd.DataFrame,
        names: list,
        list_loaded_model: list,
        exo_num_cols: list = [],
        exo_cat_cols: list = [],
    ):
        """
        Multi model approach - 1 model for 1 timestep prediction
        (tabularise data + set train valid test + prediction + concat prediction)
        i.e.
        - Prediction(t+1) = model_1(obs(t), obs(t-1), …, obs(t-w))
        - Prediction(t+2) = model_2(obs(t), obs(t-1), …, obs(t-w))
        - ...
        - Prediction(t+n) = model_n(obs(t), obs(t-1), …, obs(t-w))

        df_item_id          : a pd.DataFrame containing all item_id lvl time series
        df_calendar         : calendar.csv
        df_prices           : sell_prices.csv then agg to item_id lvl
        names               : a list of item_id name (str), len(names)=1 -> single ts
        list_loaded_model   : a sorted list of loaded lgbm models (step1-28 model)
        exo_cat_cols        : e.g. ['weekday', 'holiday'] or []
        exo_num_cols        : e.g. ['sell_price'] or []
        """

        model = list_loaded_model[0]
        windowsize = max([int(re.findall(r'\d+', feature_name)[0]) for feature_name in model.feature_name()])
        horizon = len(list_loaded_model)
        exo_num_cols = []
        exo_cat_cols = []

        # Return a df_y for validation
        df_y = (
            df_item_id[df_item_id.index.isin(names)]
            .T.iloc[-horizon:][names]
            .reset_index(drop=True)
        )

        y_pred_concat = []

        for step in np.arange(1, horizon + 1):
            print(f"---------------------Step{step}---------------------")
            df_m = TsHandler.create_training_data(
                df_item_id, names, windowsize, horizon, step=step
            )
                    
            # update index = index of y{step}
            df_m.index = df_m.index + windowsize + step - 1

            # join calendar & price
            df_m = df_m.merge(
                df_calendar[["weekday_encoded", "holiday_encoded", "wm_yr_wk"]],
                left_index=True,
                right_index=True,
                how="left",
            )

            # preserve the index for prediction
            df_m = df_m.reset_index().set_index(['item_id', 'wm_yr_wk'])

            df_m = df_m.merge(
                df_prices.set_index(['item_id', 'wm_yr_wk']),
                left_index=True,
                right_index=True,
                how="left",
            )

            # resume the index
            df_m = df_m.reset_index().set_index("index")

            # sort df_m by names
            df_m = df_m.sort_values(by='item_id', key=lambda x: x.map({name: i for i, name in enumerate(names)}))

            assert df_m.columns.isin(["weekday_encoded", "holiday_encoded", "sell_price"]).sum() == 3, "Error: join calendar & price!"

            # Create lgbm dataset for lgbm training
            num_cols = [f"x{i+1}" for i in range(windowsize)] + exo_num_cols
            target_col = f"y{step}"
            dataset_all = Dataset(
                df_m, cat_cols=exo_cat_cols, num_cols=num_cols, target_col=target_col
            )

            # Single time step prediction
            # Select the step th row of features in VALID for each item_id
            idx = min(df_m[df_m.train_valid_test=='VALID'].index) + step - 1
            features = dataset_all.num_cols + dataset_all.cat_encoded_cols
            input = df_m.loc[df_m.index == idx, features]
            y_pred = list_loaded_model[step-1].predict(input)  #shape:(len(names),)
            y_pred_concat.append(y_pred)        #len:step

        df_y_pred = pd.DataFrame(np.array(y_pred_concat), columns=names)

        return df_y, df_y_pred
    