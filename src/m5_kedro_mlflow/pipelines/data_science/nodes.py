"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

import logging
from typing import Any, Dict, Tuple

import re
import numpy as np
import pandas as pd
import lightgbm as lgbm

from m5_kedro_mlflow.pipelines.data_science.metrics import smape
from m5_kedro_mlflow.pipelines.data_science.dataset import Dataset
from m5_kedro_mlflow.pipelines.logger import Logger


def ingest_data(df, params_target, params_features, params_train_valid_test_split):
    # numerical and catergorical columns
    TARGET = params_target
    NUM_COLS = []
    CAT_COLS = []
    for i, k in params_features["num_cols"].items():
        if "num_col_" in i:
            NUM_COLS += k

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

    dataset_all = Dataset(df, CAT_COLS, NUM_COLS, TARGET)

    df_left = dataset_all.df[
        [col for col in dataset_all.df.columns if "_encoded" not in col]
    ]

    return dataset_all, df_left, labels


def lgbm_training(dataset_all, params_lgbm):
    LGBM_PARAMS = params_lgbm["lgbm_params"]
    LGBM_TRAINER_ARGS = params_lgbm["lgbm_trainer_args"]

    lgbm_model = lgbm.train(
        LGBM_PARAMS,
        **LGBM_TRAINER_ARGS,
        train_set=dataset_all.create_lgbm_dataset("TRAIN"),
        valid_sets=dataset_all.create_lgbm_dataset("VALID"),
    )

    return lgbm_model


def plot_lgbm_feature_importance(lgbm_model):
    # log
    ax_gain = lgbm.plot_importance(lgbm_model, importance_type="gain")
    ax_gain.figure.tight_layout()
    ax_split = lgbm.plot_importance(lgbm_model, importance_type="split")
    ax_split.figure.tight_layout()

    return ax_gain.figure, ax_split.figure


def prediction(lgbm_model, params_prediction, dataset_all):
    ROUND_DECIMAL = params_prediction["round_decimal"]

    y_pred = lgbm_model.predict(dataset_all.create_lgbm_dataset("ALL").data)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    if type(ROUND_DECIMAL) == int:
        y_pred = np.round(y_pred, ROUND_DECIMAL)

    df_y_pred = pd.DataFrame(columns=["y", "y_pred"])
    df_y_pred["y"] = dataset_all.create_lgbm_dataset("ALL").label
    df_y_pred["y_pred"] = y_pred
    return df_y_pred


def evaluation(df_left, df_y_pred):
    df_out = pd.concat([df_left, df_y_pred], axis=1)

    # log
    train_smape = smape(
        df_out.loc[df_out.train_valid_test == "TRAIN", "y_pred"],
        df_out.loc[df_out.train_valid_test == "TRAIN", "y"],
    )
    valid_smape = smape(
        df_out.loc[df_out.train_valid_test == "VALID", "y_pred"],
        df_out.loc[df_out.train_valid_test == "VALID", "y"],
    )
    Logger.log_metric("train_smape", train_smape)
    Logger.log_metric("valid_smape", valid_smape)
