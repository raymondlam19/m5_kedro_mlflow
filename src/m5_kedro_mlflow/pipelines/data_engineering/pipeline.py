"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_preprocess_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=read_data,
                inputs=["params:dataset"],
                outputs=["df_sales", "df_calendar", "df_prices"],
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
                outputs="de_df1",
            ),
            node(
                func=create_ma_lag_features,
                inputs=["de_df1", "params:features"],
                outputs="de_df2",
            ),
            node(
                func=final_trimming,
                inputs=["de_df2", "params:trimming"],
                outputs="de_df3",
            ),
            node(
                func=final_create_other_features,
                inputs=["de_df3", "params:features"],
                outputs="PREPROCESSED_DF",
            ),
        ]
    )
