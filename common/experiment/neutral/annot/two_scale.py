import logging

import arekit.common.experiment.neutral.annot.utils
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator

from arekit.common.experiment.neutral.annot.labels_fmt import ThreeScaleLabelsFormatter
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.experiment.data_type import DataType

logger = logging.getLogger(__name__)


class TwoScaleNeutralAnnotator(BaseNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    For two scale classification task.
    """

    def __init__(self):
        super(TwoScaleNeutralAnnotator, self).__init__(annot_name=u"neutral_2_scale")
        self.__labels_fmt = ThreeScaleLabelsFormatter()

    # region static methods

    @staticmethod
    def __iter_opinion_as_neutral(collection):
        assert(isinstance(collection, OpinionCollection))

        for opinion in collection:
            yield Opinion(source_value=opinion.SourceValue,
                          target_value=opinion.TargetValue,
                          sentiment=NeutralLabel())

    def __create_opinions_for_classification(self, doc_id):

        # TODO. Extract opinions from news.
        # TODO. exp_io.read_parsed_news.
        # TODO. exp_io.read_etalon_opinion_collection()

        collection = self._OpinOps.read_etalon_opinion_collection(doc_id)
        return self._OpinOps.create_opinon_collection(
            opinions=list(self.__iter_opinion_as_neutral(collection)))

    # endregion

    # region public methods

    def create_collection(self, data_type):
        assert(isinstance(data_type, DataType))

        if data_type == DataType.Train:
            return

        filtered_iter = self.filter_non_created_doc_ids(
            all_doc_ids=self.iter_doc_ids_to_compare(),
            data_type=data_type)

        for doc_id, filepath in filtered_iter:

            arekit.common.experiment.neutral.annot.utils.notify_newfile_creation(filepath=filepath,
                                                                                 data_type=data_type,
                                                                                 logger=logger)

            self._DataIO.OpinionFormatter.save_to_file(
                collection=self.__create_opinions_for_classification(doc_id),
                filepath=filepath,
                labels_formatter=self.__labels_fmt)

    # endregion
