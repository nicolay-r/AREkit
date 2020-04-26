import os


def check_file_already_existed(filepath, logger):
    if os.path.isfile(filepath):
        if os.path.getsize(filepath):
            logger.debug("Skipping File: {} [OK. File already exists]".format(filepath))
            return True


def notify_newfile_creation(filepath, data_type, logger):
    msg = "Create Neutral File (MODE {}): '{}'".format(data_type, filepath)
    logger.debug(msg)
