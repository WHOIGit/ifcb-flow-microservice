"""FastAPI entrypoint for the flow metric microservice."""

import os

from stateless_microservice import ServiceConfig, create_app

from .processor import FlowMetricProcessor

config = ServiceConfig(
    description="Microservice for computing flow metric anomaly scores for IFCB bins.",
)

DATA_DIR = os.getenv("DATA_DIR", "/data/ifcb")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/classifier.pkl")

app = create_app(FlowMetricProcessor(data_dir=DATA_DIR, model_path=MODEL_PATH), config)
