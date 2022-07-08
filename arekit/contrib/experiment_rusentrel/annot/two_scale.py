import logging

from arekit.common.news.parsed.base import ParsedNews
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.common.opinions.base import Opinion
from arekit.common.experiment.data_type import DataType
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel

logger = logging.getLogger(__name__)


class TwoScaleTaskOpinionAnnotator(BaseOpinionAnnotator):
    """ For two scale classification task.
    """

    def __init__(self, create_empty_collection_func, get_doc_etalon_opins_func):
        """
        create_empty_collection_func:
            function that creates an empty opinion collection
        get_doc_etalon_opinions_func:
            obtains opinion collection by a given document id
        """
        assert(callable(create_empty_collection_func))
        assert(callable(get_doc_etalon_opins_func))
        super(TwoScaleTaskOpinionAnnotator, self).__init__()

        self.__create_empty_collection_func = create_empty_collection_func
        self.__get_doc_etalon_opins_func = get_doc_etalon_opins_func

    # region static methods

    def _annot_collection_core(self, parsed_news, data_type):
        """ # TODO. #354. Remove dependency from data_type. (not used here)
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(data_type, DataType))

        doc_id = parsed_news.RelatedDocID
        neut_collection = self.__create_empty_collection_func()

        # We copy all the opinions from etalon collection
        # into neutral one with the replaced sentiment values.
        # as we treat such opinions as neutral one since only NeutralLabels
        # could be casted into correct string.
        for opinion in self.__get_doc_etalon_opins_func(doc_id):
            neut_collection.add_opinion(Opinion(source_value=opinion.SourceValue,
                                                target_value=opinion.TargetValue,
                                                sentiment=ExperimentNeutralLabel()))

        return neut_collection

    # endregion

    # region public methods

    def annotate_collection(self, data_type, parsed_news):

        if data_type == DataType.Train:
            # Return empty collection.
            return self.__create_empty_collection_func()

        super(TwoScaleTaskOpinionAnnotator, self).annotate_collection(
            data_type=data_type, parsed_news=parsed_news)

    # endregion
