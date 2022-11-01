"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_fetch_preprocess_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=read_data,
                inputs=["params:dataset"],
                outputs=["df_sales", "df_calendar", "df_prices"],
                name="read_data",
            ),
            node(
                func=trim_and_preprocess_data,
                inputs=[
                    "df_sales",
                    "df_calendar",
                    "df_prices",
                    "params:base",
                    "params:trimming",
                ],
                outputs="intermediate_df",
                name="trim_and_preprocess_data",
            ),
            node(
                func=create_lag_ma_features_and_trim_again,
                inputs=["intermediate_df", "params:trimming", "params:features"],
                outputs="preprocessed_df",
                name="create_lag_ma_features",
            ),
        ]
    )
