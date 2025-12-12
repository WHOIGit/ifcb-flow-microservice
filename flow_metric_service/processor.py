"""Flow metric anomaly score processor."""

import logging
from typing import List

import numpy as np
from pydantic import BaseModel, Field
from stateless_microservice import BaseProcessor, StatelessAction, run_blocking
from storage.utils import ReadonlyStore

from ifcb_flow_metric import FeatureExtractor, ModelTrainer, Inferencer

from .bin_store import IFCBADCStore

logger = logging.getLogger(__name__)


class FlowMetricPathParams(BaseModel):
    """Path parameters for flow metric endpoint."""

    bin_id: str = Field(..., description="IFCB bin identifier (e.g., D20120101_T120000)")


class FlowMetricProcessor(BaseProcessor):
    """Processor for computing flow metric anomaly scores."""

    def __init__(self, data_dir: str, model_path: str):
        """Initialize processor with data directory and model.

        Args:
            data_dir: Path to IFCB data directory
            model_path: Path to pre-trained classifier.pkl file
        """
        super().__init__()
        self.data_dir = data_dir
        self.model_path = model_path

        # Create ADC store
        self.adc_store = ReadonlyStore(IFCBADCStore(data_dir))

        # Load model once at initialization
        trainer = ModelTrainer(model_path)
        self.model = trainer.load_model()

        # Initialize feature extractor with config matching model (25 features, t_y_var disabled)
        feature_config = {
            'spatial_stats': {
                'mean_x': True, 'mean_y': True, 'std_x': True, 'std_y': True,
                'median_x': True, 'median_y': True, 'iqr_x': True, 'iqr_y': True
            },
            'distribution_shape': {'ratio_spread': True, 'core_fraction': True},
            'clipping_detection': {'duplicate_fraction': True, 'max_duplicate_fraction': True},
            'histogram_uniformity': {'cv_x': True, 'cv_y': True},
            'moments': {'skew_x': True, 'skew_y': True, 'kurt_x': True, 'kurt_y': True},
            'pca_orientation': {'angle': True, 'eigen_ratio': True},
            'edge_features': {
                'left_edge_fraction': True, 'right_edge_fraction': True,
                'top_edge_fraction': True, 'bottom_edge_fraction': True,
                'total_edge_fraction': True
            },
            'temporal': {'t_y_var': False}  # Disable t_y_var to match 25-feature model
        }
        self.feature_extractor = FeatureExtractor(
            aspect_ratio=1.36,
            edge_tolerance=0.05,
            feature_config=feature_config
        )

        # Initialize inferencer with loaded model
        self.inferencer = Inferencer(self.model)

        logger.info(f"FlowMetricProcessor initialized with data_dir={data_dir}, model_path={model_path}")

    @property
    def name(self) -> str:
        return "flow_metric"

    def get_stateless_actions(self) -> List[StatelessAction]:
        return [
            StatelessAction(
                name="flow_metric",
                path="/flow_metric/{bin_id}",
                path_params_model=FlowMetricPathParams,
                handler=self.handle_flow_metric,
                summary="Compute flow metric anomaly score for an IFCB bin.",
                description="Returns the anomaly score for the specified IFCB bin using pre-trained Isolation Forest model.",
                tags=("flow_metric",),
                methods=("GET",),
                media_type="text/plain",
            ),
        ]

    async def handle_flow_metric(self, path_params: FlowMetricPathParams):
        """Compute flow metric anomaly score for a bin.

        Args:
            path_params: Path parameters containing bin_id

        Returns:
            Anomaly score as string (text/plain)

        Raises:
            ValueError: If bin does not exist or data cannot be loaded
        """
        bin_id = path_params.bin_id

        # Check bin existence
        if not self.adc_store.exists(bin_id):
            raise ValueError(f"Bin {bin_id} not found in data directory")

        # Compute score in blocking context (I/O operations)
        score = await run_blocking(self._compute_score, bin_id)

        logger.info(f"Computed flow metric score for {bin_id}: {score}")

        return str(score)

    def _compute_score(self, bin_id: str) -> float:
        """Compute anomaly score for a bin (blocking operation).

        Args:
            bin_id: IFCB bin identifier

        Returns:
            Anomaly score (float)

        Raises:
            ValueError: If feature extraction or scoring fails
        """
        # Get ADC data from store
        adc = self.adc_store.get(bin_id)

        # Convert ADC DataFrame to load_result format expected by feature extractor
        points = adc[['ROI_X', 'ROI_Y']].values

        # Match get_points logic from ifcb-flow-metric
        if bin_id.startswith('I'):
            t = adc['PROCESSING_END_TIME'].values
        else:
            t = adc['ADC_TIME'].values

        load_result = {
            'pid': bin_id,
            'points': points,
            't': t
        }

        # Extract features
        feature_result = self.feature_extractor.extract_features(load_result)

        if feature_result.get('features') is None:
            raise ValueError(f"Feature extraction failed for {bin_id}")

        # Score distribution
        scores = self.inferencer.score_distributions([feature_result])

        if not scores or len(scores) == 0:
            raise ValueError(f"Scoring failed for {bin_id}")

        # Return single anomaly score
        return scores[0]['anomaly_score']
