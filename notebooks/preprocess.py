import pandas as pd
import lightgbm as lgbm


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
