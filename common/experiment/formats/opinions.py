from os.path import join
from arekit.common.experiment.data_type import DataType
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection


class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    def __init__(self):
        self.__neutral_root = None
        self._synonyms = None

    # region protected methods

    def _set_synonyms_collection(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        self._synonyms = synonyms

    def _set_neutral_root(self, neutral_root):
        assert (isinstance(neutral_root, unicode))
        self.__neutral_root = neutral_root

    # endregion

    # region abstract methods

    def read_neutral_opinion_collection(self, doc_id, data_type):
        """ data_type denotes a set of neutral opinions, where in case of 'train' these are
            opinions that were ADDITIONALLY found to sentiment, while for 'train' these are
            all the opinions that could be found in document.
        """
        raise NotImplementedError()

    def read_etalon_opinion_collection(self, doc_id):
        raise NotImplementedError()

    def get_doc_ids_set_to_neutrally_annotate(self):
        """ provides set of documents that utilized by neutral annotator algorithm in order to
            provide the related labeling of neutral attitudes in it.
            By default we consider an empty set, so there is no need to ulize neutral annotator.
        """
        raise NotImplementedError()

    def get_doc_ids_set_to_compare(self):
        """ provides a set of document ids, utilized in opinion comparison operation during
            model evaluation process.
        """
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    # TODO. This should be removed.
    # TODO. This should be removed.
    # TODO. This should be removed.
    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    # endregion

    # region public methods

    def create_opinion_collection(self, opinions=None):
        assert(isinstance(opinions, list) or opinions is None)

        if self._synonyms is None:
            raise NotImplementedError("Synonyms collection was not provided!")

        return OpinionCollection.init_as_custom(opinions=[] if opinions is None else opinions,
                                                synonyms=self._synonyms)

    # TODO. This should be removed.
    # TODO. This is used in neutral annotation in order to check the presence of
    # TODO. the related collection. It is bad way to utilize path for comparison
    # TODO. since in general we might use database to store collection.
    # TODO. Therefore we may use `check_neutral_collection_existance` method instead
    def create_neutral_opinion_collection_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        if self.__neutral_root is None:
            raise NotImplementedError("Neutral root was not provided!")

        filename = u"art{doc_id}.neut.{d_type}.txt".format(doc_id=doc_id,
                                                           d_type=data_type.name)

        return join(self.__neutral_root, filename)

    # endregion
