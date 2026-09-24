"""Unified command-line support for the UniER benchmark."""

from .registry import MODEL_SPECS, get_model_spec, model_names

__all__ = ["MODEL_SPECS", "get_model_spec", "model_names"]
