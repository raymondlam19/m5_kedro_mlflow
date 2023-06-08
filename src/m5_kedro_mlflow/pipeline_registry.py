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
    de_preprocess_pipeline = de.create_preprocess_pipeline()
    ds_lgbm_training_pipeline = ds.create_lgbm_training_pipeline("whole")

    pipelines = {
        "preprocess": de_preprocess_pipeline,
        "lgbm_training": ds_lgbm_training_pipeline,
        # "all": de_preprocess_pipeline + ds_lgbm_training_pipeline,
    }

    return pipelines
