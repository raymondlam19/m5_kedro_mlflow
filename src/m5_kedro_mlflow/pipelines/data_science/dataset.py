import lightgbm as lgbm


class Dataset:
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.df = df
        self.cat_cols = cat_cols
        self.cat_encoded_cols = [col + "_encoded" for col in cat_cols]
        self.num_cols = num_cols
        self.target_col = target_col      

    def create_lgbm_dataset(self):
        return lgbm.Dataset(
            data = self.df[self.cat_encoded_cols + self.num_cols],
            label = self.df[self.target_col],
            feature_name = self.cat_encoded_cols + self.num_cols,
            categorical_feature = self.cat_encoded_cols,
        )
