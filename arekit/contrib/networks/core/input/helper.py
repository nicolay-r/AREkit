import logging
from collections import OrderedDict

import numpy as np

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.common.experiment.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.contrib.networks.core.data.serializing import NetworkSerializationData
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.input.providers.text import NetworkSingleTextProvider
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.core.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.core.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.embeddings.base import Embedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkInputHelper(object):

    # region private methods

    @staticmethod
    def __add_term_embedding(dict_data, term, emb_vector):
        if term in dict_data:
            return
        dict_data[term] = emb_vector

    @staticmethod
    def __create_text_provider(term_embedding_pairs, exp_data, entity_to_group_func):
        assert(isinstance(exp_data, NetworkSerializationData))
        assert(isinstance(term_embedding_pairs, OrderedDict))
        assert(callable(entity_to_group_func))

        terms_with_embeddings_terms_mapper = StringWithEmbeddingNetworkTermMapping(
            entity_to_group_func=entity_to_group_func,
            predefined_embedding=exp_data.WordEmbedding,
            string_entities_formatter=exp_data.StringEntityFormatter,
            string_emb_entity_formatter=StringEntitiesSimpleFormatter())

        return NetworkSingleTextProvider(
            text_terms_mapper=terms_with_embeddings_terms_mapper,
            pair_handling_func=lambda pair: NetworkInputHelper.__add_term_embedding(
                dict_data=term_embedding_pairs,
                term=pair[0],
                emb_vector=pair[1]))

    # endregion

    @staticmethod
    def populate(opinions_target,
                 samples_target,
                 opinion_provider,
                 exp_data,
                 sample_storage,
                 opinions_storage,
                 data_type,
                 term_embedding_pairs,
                 entity_to_group_func):
        assert(isinstance(data_type, DataType))

        sample_row_provider = NetworkSampleRowProvider(
            storage=sample_storage,
            label_provider=exp_data.LabelProvider,
            text_provider=NetworkInputHelper.__create_text_provider(
                term_embedding_pairs=term_embedding_pairs,
                exp_data=exp_data,
                entity_to_group_func=entity_to_group_func),
            frames_collection=exp_data.FramesCollection,
            frame_role_label_scaler=exp_data.FrameRolesLabelScaler,
            entity_to_group_func=entity_to_group_func,
            pos_terms_mapper=PosTermsMapper(exp_data.PosTagger))
        opinion_row_provider = BaseOpinionsRowProvider()

        opinions_repo = BaseInputOpinionsRepository(
            columns_provider=None,
            rows_provider=opinion_row_provider,
            storage=opinions_storage)

        samples_repo = BaseInputOpinionsRepository(
            columns_provider=None,
            rows_provider=sample_row_provider,
            storage=sample_storage)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               target=opinions_target,
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              target=samples_target,
                              desc="sample")

    @staticmethod
    def compose_and_save_term_embeddings_and_vocabulary(experiment_io, term_embedding_pairs):
        assert(isinstance(experiment_io, NetworkIOUtils))
        assert(isinstance(term_embedding_pairs, OrderedDict))

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(
            word_embedding_pairs=iter(term_embedding_pairs.items()))

        # Save embedding matrix
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        embedding_filepath = experiment_io.get_saving_embedding_filepath()
        logger.info("Saving embedding [size={shape}]: {filepath}".format(
            shape=embedding_matrix.shape,
            filepath=embedding_filepath))
        np.savez(embedding_filepath, embedding_matrix)

        # Save vocabulary
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))
        vocab_filepath = experiment_io.get_saving_vocab_filepath()
        logger.info("Saving vocabulary [size={size}]: {filepath}".format(size=len(vocab),
                                                                         filepath=vocab_filepath))
        np.savez(vocab_filepath, vocab)

        # Remove bindings from the local namespace
        del embedding_matrix
