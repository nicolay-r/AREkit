import os


def check_file_already_existed(filepath, logger):
    if os.path.isfile(filepath):
        if os.path.getsize(filepath):
            logger.debug("Skipping File: {} [OK. File already exists]".format(filepath))
            return True

