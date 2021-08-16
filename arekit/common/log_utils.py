import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def log_synonym_existed(value):
    logger.info("Collection already has a value '{}'. Skipped".format(value.encode('utf-8')))


def log_synonym_for_entity_does_not_exist(entity_value, end_type, raise_exception):
    message = "'{s}' for end {e} does not exist in read-only SynonymsCollection".format(
        s=entity_value,
        e=end_type)

    if raise_exception:
        raise Exception(message)
    else:
        logger.info(message)


def log_opinion_already_exist(opinion, raise_exception, display_log):
    message = "'{s}->{t}' already exists in collection".format(s=opinion.SourceValue,
                                                                t=opinion.TargetValue).encode('utf-8')

    if raise_exception:
        raise Exception(message)
    elif display_log:
        logger.info(message + ' [REJECTED]')
