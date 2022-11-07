import mlflow
from kedro_mlflow.io.metrics import MlflowMetricDataSet


class Logger:
    @staticmethod
    def log_metric(key: str, metric: float):
        metric_ds = MlflowMetricDataSet(key=key)
        # with mlflow.start_run():
        metric_ds.save(
            metric
        )  # create a "my_metric=0.3" value in the "metric" field in mlflow UI
        print(f"{key}: {metric*100:.2f}%")


if __name__ == "__main__":
    Logger.log_metric("aaa", 111)
