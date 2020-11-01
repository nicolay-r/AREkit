import logging
from os.path import exists

from arekit.common.experiment.data.base import DataIO
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

    def __init__(self, data_io, experiment_io, version):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(version, RuSentRelVersions))

        super(RuSentrelOpinionOperations, self).__init__()

        self.__experiment_io = experiment_io
        self.__synonyms = data_io.SynonymsCollection
        self.__opinion_formatter = data_io.OpinionFormatter
        self.__result_labels_fmt = RuSentRelLabelsFormatter()
        self.__neutral_labels_fmt = RuSentRelNeutralLabelsFormatter()
        self.__version = version

    # region CVBasedOperations

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelOpinionCollection.load_collection(doc_id=doc_id,
                                                          synonyms=self.__synonyms,
                                                          version=self.__version)

    def create_opinion_collection(self, opinions=None):
        assert(isinstance(opinions, list) or opinions is None)

        if self.__synonyms is None:
            raise NotImplementedError("Synonyms collection was not provided!")

        return OpinionCollection.init_as_custom(opinions=[] if opinions is None else opinions,
                                                synonyms=self.__synonyms)

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
        return self.__opinion_formatter.load_from_file(filepath=filepath,
                                                       labels_formatter=labels_fmt)

    # endregion