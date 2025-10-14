"""
ClearML logger integration with credentials management.

Usage Examples:
    # Setup ClearML
    task = setup_clearml(project_name="my_project", task_name="experiment_1")
    logger = ClearMLLogger(task)
    
    # Log scalar metrics (creates unified graphs)
    for i in range(100):
        logger.log_scalar("loss", "train", 1.0/(i+1), iteration=i)
        logger.log_scalar("loss", "val", 0.5/(i+1), iteration=i)
    
    # Log images (grayscale)
    import numpy as np
    img = np.eye(256, 256, dtype=np.uint8) * 255
    logger.log_image("predictions", "sample_1", img, iteration=0)
    
    # Log RGB images
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = 255  # Red channel
    logger.log_image("predictions", "sample_rgb", img_rgb, iteration=0)
    
    # Log multiple images at once
    images = [img1, img2, img3]
    logger.log_images_batch("batch_samples", "epoch_1", images, iteration=0)
    
    # Log text
    logger.log_text("Training started successfully!")
    
    # Finalize
    logger.finalize()
"""

import os
from pathlib import Path
from omegaconf import OmegaConf
import clearml
from clearml import Task


def load_credentials(config_dir: str = None):
    """
    Load ClearML credentials from YAML file.
    
    Args:
        config_dir: Directory containing clearml_credentials.yaml
                   If None, uses the configs directory in embeddings_squeeze
    
    Returns:
        dict: Credentials dictionary
    """
    if config_dir is None:
        # Default to embeddings_squeeze/configs
        current_file = Path(__file__)
        config_dir = current_file.parent.parent / 'configs'
    else:
        config_dir = Path(config_dir)
    
    creds_file = config_dir / 'clearml_credentials.yaml'
    
    if not creds_file.exists():
        raise FileNotFoundError(
            f"ClearML credentials file not found: {creds_file}\n"
            f"Please copy clearml_credentials.yaml.example to clearml_credentials.yaml "
            f"and fill in your credentials."
        )
    
    creds = OmegaConf.load(creds_file)
    return OmegaConf.to_container(creds, resolve=True)


def setup_clearml(project_name: str, task_name: str, auto_connect: bool = True):
    """
    Setup ClearML with credentials from config file.
    
    Args:
        project_name: ClearML project name
        task_name: ClearML task name
        auto_connect: If True, automatically connect frameworks
    
    Returns:
        Task object
    """
    # Load credentials
    config_dir = Path(__file__).parent.parent / 'configs'
    
    try:
        creds = load_credentials(config_dir)
        
        # Set credentials
        clearml.Task.set_credentials(
            api_host=creds.get('api_host', 'https://api.clear.ml'),
            web_host=creds.get('web_host', 'https://app.clear.ml'),
            files_host=creds.get('files_host', 'https://files.clear.ml'),
            key=creds['api_key'],
            secret=creds['api_secret']
        )
        
        # Initialize task
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            auto_connect_frameworks=auto_connect
        )
        return task
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("ClearML logging disabled. Using TensorBoard instead.")
        return None
    except Exception as e:
        print(f"Warning: Failed to setup ClearML: {e}")
        print("ClearML logging disabled. Using TensorBoard instead.")
        return None


class ClearMLLogger:
    """
    Wrapper for ClearML logging compatible with PyTorch Lightning.
    Supports scalar metrics, plots, images, and text logging.
    """
    
    def __init__(self, task: Task):
        self.task = task
        self.logger = task.get_logger() if task else None
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to ClearML."""
        if self.logger is None:
            return
        
        for key, value in metrics.items():
            # Split key into title and series (e.g., "train/loss" -> title="train", series="loss")
            if '/' in key:
                title, series = key.split('/', 1)
            else:
                title = 'metrics'
                series = key
            
            self.logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=step
            )
    
    def log_scalar(self, title: str, series: str, value: float, iteration: int):
        """
        Log a single scalar value to ClearML.
        
        Args:
            title: Graph title (e.g., "loss", "accuracy")
            series: Series name within the graph (e.g., "train", "val")
            value: Scalar value to log
            iteration: Iteration/step number
            
        Example:
            logger.log_scalar("loss", "train", 0.5, iteration=100)
            logger.log_scalar("loss", "val", 0.3, iteration=100)
        """
        if self.logger is None:
            return
        
        self.logger.report_scalar(
            title=title,
            series=series,
            value=value,
            iteration=iteration
        )
    
    def log_image(self, title: str, series: str, image, iteration: int):
        """
        Log an image to ClearML.
        
        Args:
            title: Image title/group
            series: Series name (e.g., "predictions", "ground_truth")
            image: Image as numpy array (H, W) or (H, W, C) for grayscale/RGB
                   Supports uint8 (0-255) or float (0-1)
            iteration: Iteration/step number
            
        Example:
            # Grayscale image
            img = np.eye(256, 256, dtype=np.uint8) * 255
            logger.log_image("predictions", "epoch_1", img, iteration=0)
            
            # RGB image
            img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            img_rgb[:, :, 0] = 255  # Red channel
            logger.log_image("predictions", "epoch_1_rgb", img_rgb, iteration=0)
        """
        if self.logger is None:
            return
        
        self.logger.report_image(
            title=title,
            series=series,
            iteration=iteration,
            image=image
        )
    
    def log_images_batch(self, title: str, series: str, images: list, iteration: int):
        """
        Log multiple images to ClearML.
        
        Args:
            title: Image title/group
            series: Series name
            images: List of images (numpy arrays)
            iteration: Iteration/step number
            
        Example:
            images = [img1, img2, img3]
            logger.log_images_batch("samples", "batch_0", images, iteration=0)
        """
        if self.logger is None:
            return
        
        for idx, image in enumerate(images):
            self.logger.report_image(
                title=title,
                series=f"{series}_img_{idx}",
                iteration=iteration,
                image=image
            )
    
    def log_text(self, text: str, title: str = "Info"):
        """Log text to ClearML."""
        if self.logger is None:
            return
        self.logger.report_text(text, print_console=True)
    
    def report_text(self, text: str):
        """Report text to ClearML (alias for log_text with default title)."""
        self.log_text(text)
    
    def report_plotly(self, title: str, series: str, iteration: int, figure):
        """
        Report a Plotly figure to ClearML.
        
        Args:
            title: Plot title/group
            series: Series name
            iteration: Iteration/step number
            figure: Plotly figure object
            
        Example:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6], mode='lines', name='data'))
            fig.update_layout(title="My Plot", xaxis_title="x", yaxis_title="y")
            logger.report_plotly("metrics", "loss", iteration=0, figure=fig)
        """
        if self.logger is None:
            return
        
        self.logger.report_plotly(
            title=title,
            series=series,
            iteration=iteration,
            figure=figure
        )
    
    def finalize(self):
        """Finalize logging and close task."""
        if self.task:
            self.task.close()


class ClearMLUploadCallback:
    """
    PyTorch Lightning callback for uploading checkpoints and embeddings to ClearML.
    
    Automatically uploads:
    - Latest checkpoint after each validation epoch
    - Per-epoch validation embeddings
    
    Usage:
        from pytorch_lightning import Trainer
        from embeddings_squeeze.loggers import ClearMLUploadCallback, setup_clearml
        
        task = setup_clearml(project_name="my_project", task_name="experiment_1")
        logger = ClearMLLogger(task) if task else None
        callback = ClearMLUploadCallback(task, logger, checkpoint_dir="checkpoints")
        
        trainer = Trainer(callbacks=[callback], ...)
    """
    
    def __init__(self, task: Task, clearml_logger: ClearMLLogger = None, 
                 checkpoint_dir: str = "checkpoints", embedding_dir: str = "embeddings"):
        """
        Initialize ClearML upload callback.
        
        Args:
            task: ClearML Task object
            clearml_logger: ClearML logger for text reporting (optional)
            checkpoint_dir: Directory containing checkpoints
            embedding_dir: Directory containing embeddings
        """
        self.task = task
        self.clearml_logger = clearml_logger
        self.checkpoint_dir = checkpoint_dir
        self.embedding_dir = embedding_dir
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called after validation epoch ends."""
        if self.task is None:
            return
        
        # Upload latest checkpoint
        try:
            ckpt_path = self._find_latest_checkpoint()
            if ckpt_path:
                self.task.upload_artifact(
                    name=f"checkpoint_epoch{pl_module.current_epoch}",
                    artifact_object=ckpt_path
                )
                if self.clearml_logger:
                    self.clearml_logger.report_text(f"Uploaded checkpoint: {ckpt_path}")
        except Exception as e:
            if self.clearml_logger:
                self.clearml_logger.report_text(f"Failed uploading checkpoint: {e}")
        
        # Upload per-epoch embedding
        try:
            emb_path = os.path.join(
                self.embedding_dir, 
                f"val_embedding_epoch{pl_module.current_epoch}.pt"
            )
            if os.path.exists(emb_path):
                self.task.upload_artifact(
                    name=f"val_embedding_epoch{pl_module.current_epoch}",
                    artifact_object=emb_path
                )
                if self.clearml_logger:
                    self.clearml_logger.report_text(f"Uploaded embedding: {emb_path}")
                # Clean up local embedding file after upload
                try:
                    os.remove(emb_path)
                except Exception:
                    pass
        except Exception as e:
            if self.clearml_logger:
                self.clearml_logger.report_text(f"Failed uploading embedding: {e}")
    
    def _find_latest_checkpoint(self):
        """Find the most recently modified checkpoint file."""
        import glob
        
        # Search for checkpoints recursively
        patterns = [
            os.path.join(self.checkpoint_dir, "**/*.ckpt"),
            os.path.join(self.checkpoint_dir, "*.ckpt"),
        ]
        
        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern, recursive=True))
        
        if not matches:
            return None
        
        # Return most recently modified
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]

