"""
NFL Dataset Preprocessing Module

Handles processing of historical NFL play-by-play data (2009-2016)
and conversion to formats suitable for machine learning training.
"""

from .nfl_dataset_processor import NFLDatasetProcessor
from .training_data_builder import TrainingDataBuilder

__all__ = ["NFLDatasetProcessor", "TrainingDataBuilder"]