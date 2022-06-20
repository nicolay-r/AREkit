import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseAnnotator(object):
    """
    Performs annotation for a particular data_type
    using OpinOps and DocOps API.
    """

    def __init__(self):
        logger.info("Init annotator: [{}]".format(self.__class__))

    def _annot_collection_core(self, parsed_news, data_type):
        raise NotImplementedError

    # region public methods

    def annotate_collection(self, data_type, parsed_news):
        return self._annot_collection_core(parsed_news=parsed_news, data_type=data_type)

    # endregion
