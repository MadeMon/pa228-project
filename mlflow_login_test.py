import os

from dotenv import load_dotenv

load_dotenv()

import mlflow

mlflow.login(backend="databricks", interactive=False)

mlflow.set_experiment(f"/Users/{os.getenv('DATABRICKS_MLFLOW_USERNAME')}/test_experiment")

with mlflow.start_run():
    mlflow.log_param("test", 123)