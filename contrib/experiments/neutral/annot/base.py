import logging
import utils
from arekit.contrib.experiments.base import BaseExperiment
from arekit.common.data_type import DataType

logger = logging.getLogger(__name__)


class BaseNeutralAnnotator(object):
    """
    Performs neutral annotation for different data_type.
    """

    def __init__(self, annot_name):
        assert(isinstance(annot_name, unicode))
        self.__annot_name = annot_name
        self.__experiment = None

    @property
    def ExperimentIO(self):
        return self.__experiment

    @property
    def AnnotatorName(self):
        return self.__annot_name

    # region private methods

    def __iter_all_doc_ids(self):
        assert(isinstance(self.__experiment, BaseExperiment))
        for data_type in DataType.iter_supported():
            for doc_id in self.__experiment.iter_news_indices(data_type):
                yield doc_id

    def filter_non_created_doc_ids(self, all_doc_ids, data_type):

        for doc_id in all_doc_ids:

            filepath = self.ExperimentIO.create_neutral_opinion_collection_filepath(
                doc_id=doc_id,
                data_type=data_type)

            if utils.check_file_already_existed(filepath=filepath, logger=logger):
                continue

            yield doc_id, filepath

    # endregion

    def iter_doc_ids_to_compare(self):
        assert(isinstance(self.__experiment, BaseExperiment))
        doc_ids_iter = self.__iter_all_doc_ids()
        for doc_id in self.__experiment.iter_doc_ids_to_compare(doc_ids_iter):
            yield doc_id

    def initialize(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        self.__experiment = experiment

    def create_collection(self, data_type):
        raise NotImplementedError()


