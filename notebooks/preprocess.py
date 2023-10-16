import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    def set_train_valid_test(df, horizon):
        """
        Set train valid test for the table after a single time series df_s was tabularised

        df      : a pd.DataFrame tabularised from a single time series
        horizon : number of forecasting timestep
        """

        df["train_valid_test"] = "TRAIN"
        df["train_valid_test"].iloc[-horizon:] = "VALID"

        return df


class Dataset:
    def __init__(
        self, df: pd.DataFrame, cat_cols: list, num_cols: list, target_col: str
    ):
        self.df = df
        self.cat_cols = cat_cols
        self.cat_encoded_cols = [col + "_encoded" for col in cat_cols]
        self.num_cols = num_cols
        self.target_col = target_col

    def create_lgbm_dataset(self, train_valid_test: str):
        """
        Create a lgbm dataset for lgbm training

        train_valid_test: "TRAIN", "VALID", "TEST", "ALL"
        """

        assert train_valid_test in ("TRAIN", "VALID", "TEST", "ALL")
        if train_valid_test == "ALL":
            data = self.df.loc[:, self.cat_encoded_cols + self.num_cols]
            label = self.df.loc[:, self.target_col].rename()
        else:
            data = self.df.loc[
                self.df["train_valid_test"] == train_valid_test,
                self.cat_encoded_cols + self.num_cols,
            ]
            label = self.df.loc[
                self.df["train_valid_test"] == train_valid_test, self.target_col
            ]

        return lgbm.Dataset(
            data=data,
            label=label,
            feature_name=self.cat_encoded_cols + self.num_cols,
            categorical_feature=self.cat_encoded_cols,
        )


class Scaler:
    def scale(df_item_id: pd.DataFrame, names: list, horizon: int):
        """
        Using RobustScaler, to scale each ts by name one by one.

        df_item_id: a pd.DataFrame with item_id as index and all timesteps are the columns
        names: a list of selected item_id (str)
        horizon: horizon

        return:
        df_t_scaled: a scaled df with names as column. Shape: (training period ts len, len(selected names))
        scaler_dict: a dict {name: scaler}
        df: a df before scale for descale verification
        """
        # select those ts within TRAIN period only
        df_all = (
            df_item_id[df_item_id.index.isin(names)].T[names].reset_index(drop=True)
        )

        df_train = df_all.iloc[:-horizon]

        df_scaled = pd.DataFrame()
        scaler_dict = {}

        # fit: train, transform: all
        for name in names:
            scaler = RobustScaler()
            scaler.fit(df_train[[name]])
            df_scaled[name] = scaler.transform(df_all[[name]]).reshape(
                -1,
            )

            scaler_dict[name] = scaler

        return df_scaled, scaler_dict, df_all

    def descale(df_t_scaled: pd.DataFrame, scaler_dict: dict):
        """
        descale each scaled ts by name one by one using its own corresponding scaler which can be found in the scaler dict

        df_t_scaled: a scaled df with names as column. Shape: (training period ts len, len(selected names))
        scaler_dict: a dict {name: scaler}

        return:
        df_descaled: a descaled df with names as column. Shape: (training period ts len, len(selected names))
        """
        df_descaled = pd.DataFrame()

        for name in df_t_scaled.columns:
            df_descaled[name] = (
                scaler_dict[name]
                .inverse_transform(df_t_scaled[[name]])
                .reshape(
                    -1,
                )
            )

        return df_descaled

    def compare_2df(df1, df2):
        """
        Compare 2 df without considering the rounding errors.

        return: None
        """
        # Set a tolerance for comparing values
        tolerance = 1e-6

        # Compare the dataframes ignoring rounding errors
        comparison = np.isclose(df1, df2, atol=tolerance, equal_nan=True)

        # Check if all values are close within the tolerance
        if comparison.all():
            print("Two dataframes are equivalent, ignoring rounding errors.")
        else:
            print("Two dataframes are not equivalent, ignoring rounding errors.")
