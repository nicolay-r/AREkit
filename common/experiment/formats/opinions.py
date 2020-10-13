from os.path import join
from arekit.common.experiment.data_type import DataType
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection


class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    def __init__(self, neutral_root):
        assert(isinstance(neutral_root, unicode))
        self.__get_neutral_root = neutral_root
        self.__synonyms = None

    def _set_synonyms_collection(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        self.__synonyms = synonyms

    # region abstract methods

    def read_neutral_opinion_collection(self, doc_id, data_type):
        """ data_type denotes a set of neutral opinions, where in case of 'train' these are
            opinions that were ADDITIONALLY found to sentiment, while for 'train' these are
            all the opinions that could be found in document.
        """
        raise NotImplementedError()

    def get_doc_ids_set_to_compare(self, doc_ids):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def read_etalon_opinion_collection(self, doc_id):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    # endregion

    # region public methods

    def create_opinion_collection(self, opinions=None):
        assert(isinstance(opinions, list) or opinions is None)

        if self.__synonyms is None:
            raise NotImplementedError("Synonyms collection was not provided!")

        return OpinionCollection.init_as_custom(opinions=[] if opinions is None else opinions,
                                                synonyms=self.__synonyms)

    def create_neutral_opinion_collection_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        filename = u"art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                           d_type=data_type.name)

        return join(self.__get_neutral_root, filename)

    # endregion
