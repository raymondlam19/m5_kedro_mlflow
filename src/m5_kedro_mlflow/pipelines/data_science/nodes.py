"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgbm

from m5_kedro_mlflow.pipelines.data_science.metrics import wmape


def ingest_data(df, params_base, params_features, params_train_valid_test_split):
    # numerical and catergorical columns
    TARGET_COL = params_base["target_col"]
    NUM_COLS = []
    CAT_COLS = []
    for i, k in params_features["num_cols"].items():
        if "num_col_" in i:
            NUM_COLS += k

    if params_features["num_cols"]["ma_lag"]:
        NUM_COLS += [col for col in df.columns if TARGET_COL + "_" in col]

    for i, k in params_features["cat_cols"].items():
        if "cat_col_" in i:
            CAT_COLS += k

    # Label encoding for lgbm
    labels = {}
    for col in CAT_COLS:
        df[col + "_encoded"] = df[col].astype("category")
        label = dict(zip(df[col + "_encoded"].cat.codes, df[col + "_encoded"]))
        labels[col] = label
        df[col + "_encoded"] = df[col + "_encoded"].cat.codes

    # train valid test split
    TRAIN_D = params_train_valid_test_split["train_d"]
    VALID_D = params_train_valid_test_split["valid_d"]
    TEST_D = params_train_valid_test_split["test_d"]

    df.loc[df.d < int(TRAIN_D.split("-")[1]), "train_valid_test"] = "TRAIN"
    df.loc[
        (df.d >= int(VALID_D.split("-")[0])) & (df.d < int(VALID_D.split("-")[1])),
        "train_valid_test",
    ] = "VALID"
    df.loc[df.d >= int(TEST_D.split("-")[0]), "train_valid_test"] = "TEST"

    X_train = df.loc[
        df.train_valid_test == "TRAIN",
        [col + "_encoded" for col in CAT_COLS] + NUM_COLS,
    ]
    X_valid = df.loc[
        df.train_valid_test == "VALID",
        [col + "_encoded" for col in CAT_COLS] + NUM_COLS,
    ]
    X_test = df.loc[
        df.train_valid_test == "TEST", [col + "_encoded" for col in CAT_COLS] + NUM_COLS
    ]

    y_train = df.loc[df.train_valid_test == "TRAIN", TARGET_COL]
    y_valid = df.loc[df.train_valid_test == "VALID", TARGET_COL]
    y_test = df.loc[df.train_valid_test == "TEST", TARGET_COL]

    # log
    print(
        "Train: ",
        X_train.shape,
        y_train.shape,
        "\nValid: ",
        X_valid.shape,
        y_valid.shape,
        "\nTest: ",
        X_test.shape,
        y_test.shape,
    )

    train_set = lgbm.Dataset(
        X_train,
        y_train,
        feature_name=[col + "_encoded" for col in CAT_COLS] + NUM_COLS,
        categorical_feature=[col + "_encoded" for col in CAT_COLS],
        params={"train_valid_test": "TRAIN"},
    )

    valid_set = lgbm.Dataset(
        X_valid,
        y_valid,
        feature_name=[col + "_encoded" for col in CAT_COLS] + NUM_COLS,
        categorical_feature=[col + "_encoded" for col in CAT_COLS],
        params={"train_valid_test": "VALID"},
    )

    test_set = lgbm.Dataset(
        X_test,
        y_test,
        feature_name=[col + "_encoded" for col in CAT_COLS] + NUM_COLS,
        categorical_feature=[col + "_encoded" for col in CAT_COLS],
        params={"train_valid_test": "TEST"},
    )

    return train_set, valid_set, test_set, labels


def lgbm_training_plot_feature_importance(train_set, valid_set, params_lgbm):
    LGBM_PARAMS = params_lgbm["lgbm_params"]
    LGBM_TRAINER_ARGS = params_lgbm["lgbm_trainer_args"]

    lgbm_model = lgbm.train(
        LGBM_PARAMS,
        **LGBM_TRAINER_ARGS,
        train_set=train_set,
        valid_sets=valid_set,
    )

    # log
    lgbm.plot_importance(lgbm_model, importance_type="gain")
    lgbm.plot_importance(lgbm_model, importance_type="split")

    return lgbm_model


def prediction(lgbm_model, label_encoding_mapping_dict, *datasets):

    df_out = pd.DataFrame()
    for dataset in datasets:
        df = dataset.data.copy()

        # label decoding
        for i, k in label_encoding_mapping_dict.items():
            df[i] = df[i + "_encoded"].replace(k)

        df = df[[col for col in df.columns if "_encoded" not in col]].copy()

        # add train_valid_test col
        for i, k in dataset.params.items():
            df[i] = k

        df["y"] = dataset.label
        df["y_pred"] = lgbm_model.predict(dataset.data).clip(min=0)
        df_out = pd.concat([df_out, df], axis=0)

    return df_out


def evaluation(df_out):
    # log
    print(
        f"Train wmape: {wmape(df_out.loc[df_out.train_valid_test=='TRAIN', 'y_pred'], df_out.loc[df_out.train_valid_test=='TRAIN', 'y'])*100:.2f}%"
    )
    print(
        f"Valid wmape: {wmape(df_out.loc[df_out.train_valid_test=='VALID', 'y_pred'], df_out.loc[df_out.train_valid_test=='VALID', 'y'])*100:.2f}%"
    )
