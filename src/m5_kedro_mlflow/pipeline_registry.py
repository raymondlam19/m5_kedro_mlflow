"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import m5_kedro_mlflow.pipelines.data_engineering.pipeline as de
import m5_kedro_mlflow.pipelines.data_science.pipeline as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    de_fetch_preprocess_pipeline = de.create_fetch_preprocess_pipeline()
    ds_lgbm_model_training_pipeline = ds.create_lgbm_model_training_pipeline("whole")
    # ds_inference_pipeline = ds.create_inference_pipeline()

    pipelines = {
        "fetch_preprocess": de_fetch_preprocess_pipeline,
        "training": ds_lgbm_model_training_pipeline,
        # "ds_inference": de_pipeline + ds_inference_pipeline,
        "__default__": ds_lgbm_model_training_pipeline,
    }

    return pipelines
