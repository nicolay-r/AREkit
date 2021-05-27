import logging
from os.path import exists

from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiment_rusentrel.labels.formatters.neut_label import RuSentRelNeutralLabelsFormatter
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentrel import RuSentRelExperimentLabelsFormatter
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentrelOpinionOperations(OpinionOperations):

    def __init__(self, experiment_data, experiment_io, get_synonyms_func, version):
        assert(isinstance(experiment_data, DataIO))
        assert(isinstance(version, RuSentRelVersions))
        super(RuSentrelOpinionOperations, self).__init__()

        self.__get_synonyms_func = get_synonyms_func
        self.__version = version
        self.__experiment_io = experiment_io
        self.__opinion_formatter = experiment_data.OpinionFormatter
        self.__result_labels_fmt = RuSentRelExperimentLabelsFormatter()
        self.__neutral_labels_fmt = RuSentRelNeutralLabelsFormatter()

    # region CVBasedOperations

    def iter_opinions_for_extraction(self, doc_id, data_type):

        collections = []

        # Reading automatically annotated collection of neutral opinions.
        auto_neutral = self.try_read_annotated_opinion_collection(doc_id=doc_id,
                                                                  data_type=data_type)

        if data_type == DataType.Train:
            # Providing neutral and sentiment.
            if auto_neutral is not None:
                collections.append(auto_neutral)

            # Providing sentiment opinions.
            etalon = self.read_etalon_opinion_collection(doc_id=doc_id)
            collections.append(etalon)

        elif data_type == DataType.Test:
            # Providing neutrally labeled only
            collections.append(auto_neutral)

        for collection in collections:
            for opinion in collection:
                yield opinion

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        opins_iter = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id,
                                                                       version=self.__version)
        return self.__create_collection(opins_iter)

    def create_opinion_collection(self):
        return self.__create_collection(None)

    def try_read_annotated_opinion_collection(self, doc_id, data_type):
        filepath = self.__experiment_io.create_annotated_collection_filepath(
            doc_id=doc_id,
            data_type=data_type)

        if not exists(filepath):
            return None

        return self.__custom_read(filepath=filepath,
                                  labels_fmt=self.__neutral_labels_fmt)

    def save_annotated_opinion_collection(self, collection, doc_id, data_type):
        filepath = self.__experiment_io.create_annotated_collection_filepath(
            doc_id=doc_id,
            data_type=data_type)

        self.__opinion_formatter.save_to_file(collection=collection,
                                              filepath=filepath,
                                              labels_formatter=self.__neutral_labels_fmt)

    def read_result_opinion_collection(self, data_type, doc_id, epoch_index):
        """ Since evaluation supported only for neural networks,
            we need to guarantee the presence of a function that returns filepath
            by using isinstance command.
        """
        assert(isinstance(self.__experiment_io, BaseIOUtils))

        filepath = self.__experiment_io.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=doc_id,
            epoch_index=epoch_index)

        return self.__custom_read(filepath=filepath,
                                  labels_fmt=self.__result_labels_fmt)

    # endregion

    # region private provider methods

    def __custom_read(self, filepath, labels_fmt):
        opinions = self.__opinion_formatter.iter_opinions_from_file(filepath=filepath,
                                                                    labels_formatter=labels_fmt)

        return self.__create_collection(opinions)

    def __create_collection(self, opinions):
        return OpinionCollection(opinions=[] if opinions is None else opinions,
                                 synonyms=self.__get_synonyms_func(),
                                 error_on_duplicates=True,
                                 error_on_synonym_end_missed=True)

    # endregion