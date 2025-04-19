import os
import sys
import pandas as pd
import mlflow

working_dir = sys.argv[1]
experiment_name = sys.argv[2]
experiment_run_name = sys.argv[3]

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# Data scientist 自行定義的結果產出路徑（可根據訓練腳本來定）
output_dir = os.path.join(working_dir, "data", "train_test", "For_training_testing", "320x320", "train_test")
excel_path = os.path.join(output_dir, "training_results.xlsx")
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "default-key")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "default-secret")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name)
run_name = experiment_run_name

with mlflow.start_run(run_name=run_name):
    df_params = pd.read_excel(excel_path, sheet_name="Parameters")
    df_metrics = pd.read_excel(excel_path, sheet_name="Metrics")

    params = df_params.iloc[0].to_dict()
    for k, v in params.items():
        mlflow.log_param(k, v)

    for i, row in df_metrics.iterrows():
        mlflow.log_metrics({
            "train_accuracy": row["train_accuracy"],
            "val_accuracy": row["val_accuracy"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "train_kappa": row["train_kappa"],
            "val_kappa": row["val_kappa"]
        }, step=i)

    mlflow.log_artifact(os.path.join(output_dir, "model.h5"))
    mlflow.log_artifact(os.path.join(output_dir, "val_acc.png"))
    mlflow.log_artifact(excel_path)
