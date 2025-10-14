"""
ClearML logger integration with credentials management.
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
    
    def log_text(self, text: str, title: str = "Info"):
        """Log text to ClearML."""
        if self.logger is None:
            return
        self.logger.report_text(text, print_console=True)
    
    def finalize(self):
        """Finalize logging and close task."""
        if self.task:
            self.task.close()

