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
                    "params:trimming",
                    "params:features",
                ],
                outputs="df1",
                name="trim_and_preprocess_data",
            ),
            node(
                func=create_ma_lag_features,
                inputs=["df1", "params:features"],
                outputs="df2",
                name="create_lag_ma_features",
            ),
            node(
                func=final_trimming,
                inputs=["df2", "params:trimming"],
                outputs="df3",
                name="final_trimming",
            ),
            node(
                func=final_create_other_features,
                inputs=["df3", "params:features"],
                outputs="preprocessed_df",
                name="final_create_features",
            ),
        ]
    )
