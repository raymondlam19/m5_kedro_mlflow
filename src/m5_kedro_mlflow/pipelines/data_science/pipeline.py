"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_lgbm_training_pipeline(mode="whole") -> Pipeline:

    node_ingest = node(
        func=ingest_data,
        inputs=[
            "PREPROCESSED_DF",
            "params:target",
            "params:features",
            "params:train_valid_test_split",
        ],
        outputs=[
            "DATASET_ALL",
            "DF_LEFT",
            "LABEL_ENCODING_MAPPING_DICT",
        ],
    )

    node_train = node(
        func=lgbm_training,
        inputs=[
            "dataset_all",
            "params:lgbm",
        ],
        outputs="LGBM_TRAINED_MODEL",
    )

    node_plot_importance = node(
        func=plot_lgbm_feature_importance,
        inputs="lgbm_trained_model",
        outputs=["PLOT_FEATURE_IMPORTANCE_GAIN", "PLOT_FEATURE_IMPORTANCE_SPLIT"],
    )

    node_predict = node(
        func=prediction,
        inputs=[
            "lgbm_trained_model",
            "params:prediction",
            "dataset_all",
        ],
        outputs="DF_Y_PRED",
    )

    node_evaluate = node(
        func=evaluation,
        inputs=["DF_LEFT", "DF_Y_PRED"],
        outputs=None,
    )

    assert mode.lower() in ("whole", "train", "plot_importance", "predict", "evaluate")
    if mode.lower() == "whole":
        nodes = [
            node_ingest,
            node_train,
            node_plot_importance,
            node_predict,
            node_evaluate,
        ]
    elif mode.lower() == "train":
        nodes = [node_ingest, node_train, node_plot_importance]
    elif mode.lower() == "plot_importance":
        nodes = [node_plot_importance]
    elif mode.lower() == "predict":
        nodes = [node_predict, node_evaluate]
    elif mode.lower() == "evaluate":
        nodes = [node_evaluate]

    return pipeline(nodes)
