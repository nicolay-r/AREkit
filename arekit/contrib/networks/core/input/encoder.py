import logging
from collections import OrderedDict

import numpy as np

from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.input.provider import BaseInputProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.contrib.networks.core.data.serializing import NetworkSerializationData
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.core.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.core.input.providers.text.single import NetworkSingleTextProvider
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.core.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.embeddings.base import Embedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkInputProvider(object):

    @staticmethod
    def save(
            # TODO: remove
            opin_ops, doc_ops,
            exp_data,
            # TODO: remove exp_io
            exp_io,
            data_type,
            term_embedding_pairs,
            entity_to_group_func,
            # TODO. Remove.
            iter_parsed_news_func,
            # TODO. Remove.
            terms_per_context,
            balance):
        """
        Performs encoding for all the data_types supported by experiment.
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(exp_data, NetworkSerializationData))
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(term_embedding_pairs, OrderedDict))
        assert(isinstance(terms_per_context, int))
        assert(callable(entity_to_group_func))
        assert(isinstance(balance, bool))
        assert(callable(iter_parsed_news_func))

        # Storages.
        sample_storage = exp_io.create_samples_writer(data_type=data_type, balance=balance)
        opinions_storage = exp_io.create_opinions_writer(data_type=data_type)

        terms_with_embeddings_terms_mapper = StringWithEmbeddingNetworkTermMapping(
            entity_to_group_func=entity_to_group_func,
            predefined_embedding=exp_data.WordEmbedding,
            string_entities_formatter=exp_data.StringEntityFormatter,
            string_emb_entity_formatter=StringEntitiesSimpleFormatter())

        text_provider = NetworkSingleTextProvider(
            text_terms_mapper=terms_with_embeddings_terms_mapper,
            pair_handling_func=lambda pair: NetworkInputProvider.__add_term_embedding(
                dict_data=term_embedding_pairs,
                term=pair[0],
                emb_vector=pair[1]))

        opinion_provider = OpinionProvider.from_experiment(
            doc_ops=doc_ops,
            opin_ops=opin_ops,
            data_type=data_type,
            parsed_news_it_func=iter_parsed_news_func,
            terms_per_context=terms_per_context)

        # Providers.
        sample_row_provider = NetworkSampleRowProvider(
            storage=sample_storage,
            label_provider=exp_data.LabelProvider,
            text_provider=text_provider,
            frames_collection=exp_data.FramesCollection,
            frame_role_label_scaler=exp_data.FrameRolesLabelScaler,
            entity_to_group_func=entity_to_group_func,
            pos_terms_mapper=PosTermsMapper(exp_data.PosTagger))
        opinion_row_provider = BaseOpinionsRowProvider(storage=opinions_storage)

        # Encoding input
        BaseInputProvider.save(
            opinion_provider=opinion_provider,
            opinion_row_provider=opinion_row_provider,
            sample_row_provider=sample_row_provider)

    @staticmethod
    def __add_term_embedding(dict_data, term, emb_vector):
        if term in dict_data:
            return
        dict_data[term] = emb_vector

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
