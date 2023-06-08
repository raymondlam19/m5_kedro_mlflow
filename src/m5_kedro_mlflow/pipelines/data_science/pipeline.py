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
            "preprocessed_df",
            "params:target",
            "params:features",
            "params:train_valid_test_split",
        ],
        outputs=[
            "dataset_all",
            "df_left",
            "label_encoding_mapping_dict",
        ],
    )

    node_train = node(
        func=lgbm_training,
        inputs=[
            "dataset_all",
            "params:lgbm",
        ],
        outputs="lgbm_trained_model",
    )

    node_plot_importance = node(
        func=plot_lgbm_feature_importance,
        inputs="lgbm_trained_model",
        outputs=["plot_feature_importance_gain", "plot_feature_importance_split"],
    )

    node_predict = node(
        func=prediction,
        inputs=[
            "lgbm_trained_model",
            "params:prediction",
            "dataset_all",
        ],
        outputs="df_y_pred",
    )

    node_evaluate = node(
        func=evaluation,
        inputs=["df_left", "df_y_pred"],
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
