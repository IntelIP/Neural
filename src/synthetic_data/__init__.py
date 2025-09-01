"""
Synthetic Football Data Generation System

A comprehensive system for generating realistic NFL game scenarios
using historical data and fine-tuned language models.

Components:
- preprocessing: NFL dataset processing and formatting
- generators: Game sequence and scenario generation
- models: Fine-tuned LFM2 model wrappers
- storage: ChromaDB integration and data management
- validation: Quality metrics and pattern analysis
"""

__version__ = "1.0.0"
__author__ = "Neural Trading Platform"

from . import preprocessing, generators, models, storage, validation

__all__ = ["preprocessing", "generators", "models", "storage", "validation"]