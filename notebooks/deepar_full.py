import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import statsmodels.api as sm

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import torch

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import (
    MAE,
    SMAPE,
    MultivariateNormalDistributionLoss,
    NormalDistributionLoss,
)
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy

from evaluation import Metric, Evaluation
from plotting import Plot, ModelEvaluation
from time_series import TsHandler
from preprocess import Preprocessor, Dataset, Scaler
from multi_horizon import Prediction
from config.get_config import ConfigHandler, HOME_PATH

import warnings

warnings.filterwarnings("ignore")

import random

seed = 42


def read_data():
    print("----------------1.read_data--------------")
    df_sales = pd.read_csv(f"{HOME_PATH}/notebooks/data/sales_train_evaluation.csv")

    # agg to item lvl for obtaining multi time series with obvious patterns
    df = df_sales.drop(
        columns=[
            "id",
            # 'item_id',
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ]
    )
    df_item_id = df.groupby(["item_id"]).sum()

    unpivot = pd.melt(
        df_item_id.T.reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "time_idx"}),
        id_vars="time_idx",
        var_name="item_id",
    )
    unpivot["category"] = unpivot["item_id"].str.split("_").str[0]

    # It is important to set "value" from int to float
    # Otherwises, KeyError: "Unknown category '923' encountered. Set `add_nan=True` to allow unknown categories"
    unpivot = unpivot.astype({"value": "float64"})
    return unpivot


def create_dataset(unpivot, context_length, prediction_length):
    print("----------------2.create_dataset--------------")
    # create dataset and dataloaders
    # context_length = WINDOWSIZE
    # prediction_length = HORIZON
    training_cutoff = unpivot["time_idx"].max() - prediction_length

    unpivot_training_set = unpivot[unpivot["time_idx"] <= training_cutoff]
    # unpivot_valid_set = unpivot[unpivot["time_idx"] > training_cutoff]
    encoder = NaNLabelEncoder(add_nan=True).fit(unpivot_training_set["category"])

    train_dataset = TimeSeriesDataSet(
        unpivot_training_set,
        time_idx="time_idx",
        target="value",
        categorical_encoders={"category": encoder},
        group_ids=["item_id"],
        static_categoricals=["category"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        # static_reals=[ ... ],
        # time_varying_known_categoricals=[ ... ],
        # time_varying_known_reals=[ ... ],
        # time_varying_unknown_categoricals=[ ... ],
        # time_varying_unknown_reals=[ ... ],
        # variable_groups
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, unpivot, min_prediction_idx=training_cutoff + 1
    )

    return train_dataset, val_dataset


def create_dataloader(train_dataset, val_dataset):
    print("----------------3.create_dataloader--------------")
    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=4,
        # batch_sampler="synchronized"    # for DEEPVAR
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=4,
        # batch_sampler="synchronized"    # for DEEPVAR
    )

    return train_dataloader, val_dataloader


def create_model_architecture(train_dataset):
    print("----------------4.create_model_architecture--------------")
    model = DeepAR.from_dataset(
        train_dataset,
        learning_rate=1e-2,
        hidden_size=1024,
        rnn_layers=2,
        loss=NormalDistributionLoss(),  # MultivariateNormalDistributionLoss(rank=30),   # for DEEPVAR
        optimizer="Adam",
    )

    return model


def find_learning_rate(model, train_dataloader, val_dataloader):
    print("----------------5.find_learning_rate--------------")
    pl.seed_everything(42)

    # Explicitly specify the process group backend if you choose to
    strategy = DDPStrategy(
        process_group_backend="gloo"
    )  # DDPStrategy(process_group_backend="gloo")  # DeepSpeedStrategy(process_group_backend="gloo")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0, 1],
        strategy=strategy,
        gradient_clip_val=0.1,
    )
    res = Tuner(trainer).lr_find(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-4,
        max_lr=1e-2,
        num_training=300,
    )
    fig = res.plot(show=True, suggest=True)
    fig.show()
    print(f"suggested learning rate: {res.suggestion()}")
    return res.suggestion()  # 2048,2: 0.00012882495516931342


def training(model, train_dataloader, val_dataloader, epochs=10, optimal_lr=None):
    print("----------------6.training--------------")
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    strategy = DDPStrategy(process_group_backend="gloo")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[0, 1],
        strategy=strategy,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=f"{HOME_PATH}/notebooks/saved/model/",
    )
    if optimal_lr:
        model.hparams.learning_rate = optimal_lr

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")

    return trainer


def load_best_model(trainer):
    print("----------------7.load best model--------------")
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = DeepAR.load_from_checkpoint(best_model_path)
    return best_model


def find_smape(predictions):
    print("----------------find smape--------------")
    smape = SMAPE()(predictions.output, predictions.y).cpu().numpy()
    print(f"smape: {smape}")

    return smape

def predict(model, val_dataloader):
    print("----------------8.Predict--------------")
    predictions = model.predict(val_dataloader, trainer_kwargs=dict(accelerator="gpu", strategy = DDPStrategy(process_group_backend="gloo")), return_y=True)
    return predictions


def predict_raw(model, val_dataloader):
    print("----------------8.Predict raw--------------")
    raw_predictions = model.predict(
        val_dataloader,
        mode="raw",
        return_x=True,
        n_samples=100,
        trainer_kwargs=dict(accelerator="gpu"),
    )
    return raw_predictions


def evaluate(best_model, raw_predictions, predictions):
    # TODO
    pass


if __name__ == "__main__":
    # Read model_params from notebooks/config/model_params.yml
    batch_size = 256

    model_params = ConfigHandler.read_yml("model_params_path")

    HORIZON = model_params["common"]["horizon"]
    WINDOWSIZE = HORIZON * 3  # model_params['common']['windowsize']

    print(f"windowsize:{WINDOWSIZE}")
    print(f"horizon:{HORIZON}")
    print(f"HOME_PATH:{HOME_PATH}")

    unpivot = read_data()
    train_dataset, val_dataset = create_dataset(
        unpivot, context_length=WINDOWSIZE, prediction_length=HORIZON
    )
    train_dataloader, val_dataloader = create_dataloader(train_dataset, val_dataset)
    model = create_model_architecture(train_dataset)

    baseline_predictions = predict(Baseline(), val_dataloader)
    # baseline_smape = find_smape(baseline_predictions)

    # optimal_lr = find_learning_rate(model, train_dataloader, val_dataloader)

    trainer = training(
        model, train_dataloader, val_dataloader, epochs=10, optimal_lr=0.00012208622540590714
    )

    best_model = load_best_model(trainer)
    predictions = predict(best_model, val_dataloader)
    smape = find_smape(predictions)
