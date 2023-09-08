import lightgbm as lgbm


class Preprocessor:
    def set_train_valid_test(df, horizon):
        # train valid test split
        df["train_valid_test"] = "TRAIN"
        df["train_valid_test"].iloc[-horizon:] = "VALID"

        return df


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


# if __name__=="__main__":
#     # unit test
#     aaa = Preprocessor.set_train_valid_test(df2, horizon=HORIZON)
#     aaa[aaa['train_valid_test']=="VALID"]
