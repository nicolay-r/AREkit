import glob
import logging
import os
import shutil


def setup_logger():
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


def rm_dir_contents(dir_path, logger):
    contents = glob.glob(dir_path)
    for f in contents:
        logger.info("Removing old file/dir: {}".format(f))
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f, ignore_errors=True)
