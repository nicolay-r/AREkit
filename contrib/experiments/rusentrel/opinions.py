import collections
import logging
from os.path import exists

from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.rusentrel.labels_formatter import RuSentRelNeutralLabelsFormatter
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentrelOpinionOperations(OpinionOperations):

    def __init__(self, experiment_data, experiment_io, synonyms, version):
        assert(isinstance(experiment_data, DataIO))
        assert(isinstance(version, RuSentRelVersions))

        super(RuSentrelOpinionOperations, self).__init__(synonyms)

        self.__version = version
        self.__experiment_io = experiment_io
        self.__opinion_formatter = experiment_data.OpinionFormatter
        self.__result_labels_fmt = RuSentRelLabelsFormatter()
        self.__neutral_labels_fmt = RuSentRelNeutralLabelsFormatter()

    # region CVBasedOperations

    def iter_opinions_for_extraction(self, doc_id, data_type):

        # Reading automatically annotated collection of neutral opinions.
        auto_neutral = self.try_read_neutral_opinion_collection(doc_id=doc_id,
                                                                data_type=data_type)
        if data_type == DataType.Train:
            # Providing neutral and sentiment.
            if auto_neutral is not None:
                yield iter(auto_neutral)

            # Providing sentiment opinions.
            yield self.read_etalon_opinion_collection(doc_id=doc_id)

        elif data_type == DataType.Test:
            # Providing neutrally labeled only
            yield iter(auto_neutral)

        # Provide nothing otherwise
        pass

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        opins_iter = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id,
                                                                       version=self.__version)

        return OpinionCollection(opinions=opins_iter,
                                 synonyms=self.SynonymsCollection,
                                 error_on_duplicates=True,
                                 error_on_synonym_end_missed=True)

    def create_opinion_collection(self, opinions=None):
        assert(isinstance(opinions, collections.Iterable) or opinions is None)
        return self.__create_custom_collection(opinions)

    def try_read_neutral_opinion_collection(self, doc_id, data_type):
        filepath = self.__experiment_io.create_neutral_opinion_collection_filepath(
            doc_id=doc_id,
            data_type=data_type)

        if not exists(filepath):
            return None

        return self.__custom_read(filepath=filepath,
                                  labels_fmt=self.__neutral_labels_fmt)

    def save_neutral_opinion_collection(self, collection, labels_fmt, doc_id, data_type):
        filepath = self.__experiment_io.create_neutral_opinion_collection_filepath(
            doc_id=doc_id,
            data_type=data_type)

        self.__opinion_formatter.save_to_file(collection=collection,
                                              filepath=filepath,
                                              labels_formatter=labels_fmt)

    def read_result_opinion_collection(self, data_type, doc_id, epoch_index):
        """ Since evaluation supported only for neural networks,
            we need to gaurantee the presence of a function that returns filepath
            by using isisntance command.
        """
        assert(isinstance(self.__experiment_io, NetworkIOUtils))

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

        return self.__create_custom_collection(opinions)

    def __create_custom_collection(self, opinions):
        return OpinionCollection.init_as_custom(opinions=[] if opinions is None else opinions,
                                                synonyms=self.SynonymsCollection)

    # endregion