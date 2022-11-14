"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgbm

from m5_kedro_mlflow.pipelines.data_science.metrics import smape
from m5_kedro_mlflow.pipelines.data_science.dataset import Dataset
from m5_kedro_mlflow.pipelines.logger import Logger


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

    dataset_train = Dataset(
        df[df.train_valid_test == "TRAIN"], CAT_COLS, NUM_COLS, TARGET_COL
    )
    dataset_valid = Dataset(
        df[df.train_valid_test == "VALID"], CAT_COLS, NUM_COLS, TARGET_COL
    )
    dataset_test = Dataset(
        df[df.train_valid_test == "TEST"], CAT_COLS, NUM_COLS, TARGET_COL
    )

    return dataset_train, dataset_valid, dataset_test, labels


def lgbm_training(dataset_train, dataset_valid, params_lgbm):
    LGBM_PARAMS = params_lgbm["lgbm_params"]
    LGBM_TRAINER_ARGS = params_lgbm["lgbm_trainer_args"]

    lgbm_model = lgbm.train(
        LGBM_PARAMS,
        **LGBM_TRAINER_ARGS,
        train_set=dataset_train.create_lgbm_dataset(),
        valid_sets=dataset_valid.create_lgbm_dataset(),
    )

    return lgbm_model


def plot_lgbm_feature_importance(lgbm_model):
    # log
    ax_gain = lgbm.plot_importance(lgbm_model, importance_type="gain")
    ax_gain.figure.tight_layout()
    ax_split = lgbm.plot_importance(lgbm_model, importance_type="split")
    ax_split.figure.tight_layout()

    return ax_gain.figure, ax_split.figure


def prediction(lgbm_model, *datasets):
    cat_encoded_cols = datasets[0].cat_encoded_cols
    num_cols = datasets[0].num_cols
    target_col = datasets[0].target_col

    df_out = pd.concat([dataset.df for dataset in datasets], axis=0)

    df_out["y_pred"] = lgbm_model.predict(df_out[cat_encoded_cols + num_cols]).clip(
        min=0
    )

    df_out = df_out[[col for col in df_out.columns if "_encoded" not in col]].rename(
        columns={target_col: "y"}
    )

    return df_out


def evaluation(df_out):
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
