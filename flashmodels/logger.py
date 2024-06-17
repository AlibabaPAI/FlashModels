import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.propagate = False
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
