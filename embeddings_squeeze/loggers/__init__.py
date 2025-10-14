"""Logger integrations for embeddings_squeeze."""

from .clearml_logger import setup_clearml, ClearMLLogger, ClearMLUploadCallback

__all__ = ['setup_clearml', 'ClearMLLogger', 'ClearMLUploadCallback']

