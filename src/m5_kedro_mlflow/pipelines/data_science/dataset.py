import lightgbm as lgbm


class Dataset:
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.df = df
        self.cat_cols = cat_cols
        self.cat_encoded_cols = [col + "_encoded" for col in cat_cols]
        self.num_cols = num_cols
        self.target_col = target_col

    def create_lgbm_dataset(self, train_valid_test):
        assert train_valid_test in ("TRAIN", "VALID", "TEST", "ALL")
        if train_valid_test == "ALL":
            data = self.df.loc[:, self.cat_encoded_cols + self.num_cols]
            label = self.df.loc[:, self.target_col]
        else:
            data = self.df.loc[
                self.df["train_valid_test"] == train_valid_test,
                self.cat_encoded_cols + self.num_cols,
            ]
            label = self.df.loc[
                self.df["train_valid_test"] == train_valid_test, self.target_col
            ]

        return lgbm.Dataset(
            data = data,
            label = label,
            feature_name = self.cat_encoded_cols + self.num_cols,
            categorical_feature = self.cat_encoded_cols,
        )
