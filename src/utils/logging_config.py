import logging
import sys
import os


output_dir = os.getenv("OUTPUT_DIR", "")

# Configure the logger
log_file = "project.log"
log_file = os.path.join(output_dir, "BEAST.log")


def setup_logger():
    logger = logging.getLogger("BEAST-logger")
    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers if re-imported

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
