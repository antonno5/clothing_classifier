from typing import Any, Dict

import pytorch_lightning as pl


def get_logger(logging_conf: Dict[str, Any]) -> pl.loggers.logger.Logger:
    return pl.loggers.MLFlowLogger(
        experiment_name=logging_conf["experiment_name"],
        run_name=logging_conf["run_name"],
        save_dir=logging_conf["mlflow_save_dir"],
        tracking_uri=logging_conf["tracking_uri"],
    )
