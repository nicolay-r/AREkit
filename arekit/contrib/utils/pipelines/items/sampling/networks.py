from arekit.contrib.networks.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.embedding import Embedding
from arekit.contrib.networks.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem


class NetworksInputSerializerPipelineItem(BaseSerializerPipelineItem):

    def __init__(self, save_labels_func, rows_provider, samples_io, emb_io, storage, save_embedding=True):
        """ This pipeline item allows to perform a data preparation for neural network models.

            considering a list of the whole data_types with the related pipelines,
            which are supported and required in a handler. It is necessary to know
            data_types in advance as it allows to create a complete vocabulary of input terms,
            with the related embeddings.
        """
        assert(isinstance(emb_io, NpEmbeddingIO))
        assert(isinstance(rows_provider, NetworkSampleRowProvider))
        assert(isinstance(save_embedding, bool))
        super(NetworksInputSerializerPipelineItem, self).__init__(
            rows_provider=rows_provider,
            samples_io=samples_io,
            save_labels_func=save_labels_func,
            storage=storage)

        self.__emb_io = emb_io
        self.__save_embedding = save_embedding

    def _handle_iteration(self, data_type_pipelines, data_folding, doc_ids):
        """ Performing data serialization for a particular iteration
        """
        assert(isinstance(data_type_pipelines, dict))

        # Prepare for the present iteration.
        self._rows_provider.clear_embedding_pairs()

        super(NetworksInputSerializerPipelineItem, self)._handle_iteration(
            data_type_pipelines=data_type_pipelines, data_folding=data_folding, doc_ids=doc_ids)

        if not (self.__save_embedding and self._rows_provider.HasEmbeddingPairs):
            return

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(self._rows_provider.iter_term_embedding_pairs())
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))

        # Save embedding matrix
        self.__emb_io.save_embedding(data=embedding_matrix)
        self.__emb_io.save_vocab(data=vocab)

        del embedding_matrix
