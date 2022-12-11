"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_lgbm_model_training_pipeline(mode="whole", **kwargs) -> Pipeline:

    node_get_target = node(
        func=get_target,
        inputs=[
            "preprocessed_df",
            "params:target",
        ],
        outputs="df1",
        name="get_target",
    )

    node_ingest = node(
        func=ingest_data,
        inputs=[
            "df1",
            "params:target",
            "params:features",
            "params:train_valid_test_split",
        ],
        outputs=[
            "dataset_all",
            "df_left",
            "label_encoding_mapping_dict",
        ],
        name="ingest_data",
    )

    node_train = node(
        func=lgbm_training,
        inputs=[
            "dataset_all",
            "params:lgbm",
        ],
        outputs="lgbm_trained_model",
        name="lgbm_training",
    )

    node_plot_importance = node(
        func=plot_lgbm_feature_importance,
        inputs="lgbm_trained_model",
        outputs=["plot_feature_importance_gain", "plot_feature_importance_split"],
        name="plot_lgbm_feature_importance",
    )

    node_predict = node(
        func=prediction,
        inputs=[
            "lgbm_trained_model",
            "params:prediction",
            "dataset_all",
        ],
        outputs="df_y_pred",
        name="prediction",
    )

    # TODO3
    # node_postprocess = node(
    #     func=postprocess,
    #     inputs=[
    #         "lgbm_trained_model",
    #         "params:prediction",
    #         "dataset_all",
    #     ],
    #     outputs="y_pred",
    #     name="prediction",
    # )

    node_evaluate = node(
        func=evaluation,
        inputs=["df_left", "df_y_pred"],
        outputs=None,
        name="evaluation",
    )

    assert mode.lower() in ("whole", "train", "plot_importance", "predict", "evaluate")
    if mode.lower() == "whole":
        nodes = [
            node_get_target,
            node_ingest,
            node_train,
            node_plot_importance,
            node_predict,
            node_evaluate,
        ]
    elif mode.lower() == "train":
        nodes = [node_get_target, node_ingest, node_train, node_plot_importance]
    elif mode.lower() == "plot_importance":
        nodes = [node_plot_importance]
    elif mode.lower() == "predict":
        nodes = [node_predict, node_evaluate]
    elif mode.lower() == "evaluate":
        nodes = [node_evaluate]

    return pipeline(nodes)
