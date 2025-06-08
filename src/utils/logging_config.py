import logging
import os
import sys

output_dir = os.getenv("OUTPUT_DIR", "")

# Configure the logger
log_file = "project.log"
log_file = os.path.join(output_dir, "BEAST.log")


# This is partially from ChatGPT
def setup_logger():
    logger = logging.getLogger("BEAST-logger")
    logger.propagate = False  # Prevent messages from being passed to the root logger

    # Check if the logger already has handlers to avoid duplication
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Create file and console handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stderr)

    # Set logging levels for each handler
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Define a common log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger for the entire project
logger = setup_logger()
