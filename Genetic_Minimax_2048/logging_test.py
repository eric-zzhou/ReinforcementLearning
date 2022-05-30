import logging
import sys

# Logging stuff
formatter = logging.Formatter(fmt='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TrainingFileLog")
file_handler = logging.FileHandler(filename='test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.WARNING)
logger2 = logging.getLogger("TrainingConsoleLog")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger2.addHandler(stream_handler)
logger2.setLevel(logging.INFO)

logger.debug("debug test")
logger2.debug("debug test")
logger.info("info test")
logger2.info("info test")
logger.warning("warning test")
logger2.warning("warning test")
logger.error("error test")
logger2.error("error test")
logger.critical("critical test")
logger2.critical("critical test")
