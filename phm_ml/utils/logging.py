import logging
from typing import Optional, Union, Literal
from pathlib import Path
import sys

def setup_logging(
    name: str = "phm_ml",
    log_file: Optional[Union[str, Path]] = None,
    level: Union[int, Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = "INFO",
    file_mode: str = "a"
) -> logging.Logger:
    """Set up a simple logger with console and optional file output.
    
    Args:
        name: Logger name. Defaults to "phm_ml".
        log_file: Optional path to log file. If None, logs only to console.
        level: Logging level, either as string name or logging constant.
        file_mode: File opening mode. Defaults to append ("a").
    
    Returns:
        Configured logger instance.
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level)
    
    # Create logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        if isinstance(log_file, str):
            log_file = Path(log_file)
        
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger