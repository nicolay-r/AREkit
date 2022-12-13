from os.path import join

from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.networks.embedding_io import BaseEmbeddingIO
from arekit.contrib.utils.io_utils.utils import check_targets_existence
from arekit.contrib.utils.np_utils.embedding import NpzEmbeddingHelper
from arekit.contrib.utils.np_utils.vocab import VocabRepositoryUtils
from arekit.contrib.utils.utils_folding import experiment_iter_index


class NpEmbeddingIO(BaseEmbeddingIO):
    """ Npz-based IO utils for embedding and text-based for vocabulary.
        This format represents a archived version of the numpy math data, i.e. vectors, numbers, etc.

        Provides additional Input/Output paths generation functions for:
            - embedding matrix;
            - embedding vocabulary.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = 'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = "vocab-{cv_index}.txt"

    def __init__(self, target_dir):
        assert(isinstance(target_dir, str))
        self.__target_dir = target_dir

    # region Embedding-related data

    def save_vocab(self, data, data_folding):
        assert(isinstance(data_folding, BaseDataFolding))
        target = self.__get_default_vocab_filepath(data_folding)
        return VocabRepositoryUtils.save(data=data, target=target)

    def load_vocab(self, data_folding):
        source = self.___get_vocab_source(data_folding)
        return dict(VocabRepositoryUtils.load(source))

    def save_embedding(self, data, data_folding):
        assert(isinstance(data_folding, BaseDataFolding))
        target = self.__get_default_embedding_filepath(data_folding)
        NpzEmbeddingHelper.save_embedding(data=data, target=target)

    def load_embedding(self, data_folding):
        source = self.__get_term_embedding_source(data_folding)
        return NpzEmbeddingHelper.load_embedding(source)

    def check_targets_existed(self, data_folding):
        targets = [
            self.__get_default_vocab_filepath(data_folding=data_folding),
            self.__get_term_embedding_target(data_folding=data_folding)
        ]
        return check_targets_existence(targets=targets)

    # endregion

    # region embedding-related data

    def ___get_vocab_source(self, data_folding):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        return self.__get_default_vocab_filepath(data_folding)

    def __get_term_embedding_target(self, data_folding):
        return self.__get_default_embedding_filepath(data_folding)

    def __get_term_embedding_source(self, data_folding):
        return self.__get_default_embedding_filepath(data_folding)

    def __get_default_vocab_filepath(self, data_folding):
        return join(self.__target_dir,
                    self.VOCABULARY_FILENAME_TEMPLATE.format(
                        cv_index=experiment_iter_index(data_folding)))

    def __get_default_embedding_filepath(self, data_folding):
        return join(self.__target_dir,
                    self.TERM_EMBEDDING_FILENAME_TEMPLATE.format(
                        cv_index=experiment_iter_index(data_folding)) + '.npz')

    # endregion
