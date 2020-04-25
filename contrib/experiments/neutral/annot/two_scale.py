import logging

import utils
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.data_type import DataType

logger = logging.getLogger(__name__)


class TwoScaleNeutralAnnotator(BaseNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    For two scale classification task.
    """

    def __init__(self):
        super(TwoScaleNeutralAnnotator, self).__init__(
            annot_name=u"neutral_2_scale")

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

        news, _ = self.Experiment.read_parsed_news(doc_id)
        collection = self.Experiment.read_etalon_opinion_collection(doc_id)

        return self.Experiment.create_opinon_collection(
            opinions=list(self.__iter_opinion_as_neutral(collection)))

    # endregion

    # region public methods

    def create_collection(self, data_type):
        assert(isinstance(data_type, unicode))

        if data_type == DataType.Train:
            return

        filtered_iter = self.filter_non_created_doc_ids(
            all_doc_ids=self.iter_doc_ids_to_compare(),
            data_type=data_type)

        for doc_id, filepath in filtered_iter:

            utils.notify_newfile_creation(filepath=filepath,
                                          data_type=data_type,
                                          logger=logger)

            self.Experiment.DataIO.OpinionFormatter.save_to_file(
                collection=self.__create_opinions_for_classification(doc_id),
                filepath=filepath)

    # endregion
