import glob
import logging
import os
import shutil
from os.path import join, exists
from arekit.common.utils import create_dir_if_not_exists


def mark_dir_for_serialization(target_dir, logger, skip_if_folder_exists):
    assert(isinstance(target_dir, unicode))
    assert(isinstance(skip_if_folder_exists, bool))

    target_file = join(target_dir, 'lock.txt')
    if exists(target_file) and skip_if_folder_exists:
        logger.info("TARGET DIR EXISTS: {}".format(target_dir))
        return
    else:
        create_dir_if_not_exists(target_dir)
        open(target_file, 'a').close()


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
