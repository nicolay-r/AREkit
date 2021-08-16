import logging
from collections import OrderedDict

import numpy as np

from arekit.common.embeddings.base import Embedding
from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.contrib.networks.core.data.serializing import NetworkSerializationData
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.core.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.core.input.formatters.sample import NetworkSampleFormatter
from arekit.contrib.networks.core.input.providers.text.single import NetworkSingleTextProvider
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.core.input.embedding.matrix import create_term_embedding_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkInputEncoder(object):

    @staticmethod
    def to_tsv_with_embedding_and_vocabulary(opin_ops, doc_ops, exp_data, exp_io,
                                             data_type, term_embedding_pairs, entity_to_group_func,
                                             iter_parsed_news_func, terms_per_context, balance):
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

        terms_with_embeddings_terms_mapper = StringWithEmbeddingNetworkTermMapping(
            entity_to_group_func=entity_to_group_func,
            predefined_embedding=exp_data.WordEmbedding,
            string_entities_formatter=exp_data.StringEntityFormatter,
            string_emb_entity_formatter=StringEntitiesSimpleFormatter())

        text_provider = NetworkSingleTextProvider(
            text_terms_mapper=terms_with_embeddings_terms_mapper,
            pair_handling_func=lambda pair: NetworkInputEncoder.__add_term_embedding(
                dict_data=term_embedding_pairs,
                term=pair[0],
                emb_vector=pair[1]))

        # Encoding input
        BaseInputEncoder.to_tsv(
            sample_filepath=exp_io.get_input_sample_filepath(data_type=data_type),
            opinion_filepath=exp_io.get_input_opinions_filepath(data_type=data_type),
            opinion_formatter=BaseOpinionsFormatter(data_type),
            opinion_provider=OpinionProvider.from_experiment(
                doc_ops=doc_ops,
                opin_ops=opin_ops,
                data_type=data_type,
                parsed_news_it_func=iter_parsed_news_func,
                terms_per_context=terms_per_context),
            sample_formatter=NetworkSampleFormatter(
                data_type=data_type,
                label_provider=exp_data.LabelProvider,
                text_provider=text_provider,
                frame_role_label_scaler=exp_data.FrameRolesLabelScaler,
                entity_to_group_func=entity_to_group_func,
                frames_collection=exp_data.FramesCollection,
                balance=balance and data_type == DataType.Train,
                pos_terms_mapper=PosTermsMapper(exp_data.PosTagger)),
            write_sample_header=True)

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
