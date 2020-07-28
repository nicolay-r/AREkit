import logging

import numpy as np

from arekit.common.embeddings.base import Embedding
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.contrib.networks.entities.str_emb_fmt import StringWordEmbeddingEntityFormatter
from arekit.networks.input.embedding.matrix import create_term_embedding_matrix
from arekit.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.networks.input.formatters.sample import NetworkSampleFormatter
from arekit.networks.input.providers.text.single import NetworkSingleTextProvider
from arekit.networks.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.networks.io_utils import NetworkIOUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkInputEncoder(object):

    @staticmethod
    def to_tsv_with_embedding_and_vocabulary(experiment, terms_per_context):
        """
        Performs encodding for all the data_types supported by experiment.
        """
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))

        predefined_embedding = experiment.DataIO.WordEmbedding
        predefined_embedding.set_stemmer(experiment.DataIO.Stemmer)

        terms_with_embeddings_terms_mapper = StringWithEmbeddingNetworkTermMapping(
            synonyms=experiment.DataIO.SynonymsCollection,
            predefined_embedding=predefined_embedding,
            string_entities_formatter=experiment.DataIO.StringEntityFormatter,
            string_emb_entity_formatter=StringWordEmbeddingEntityFormatter())

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type)

        term_embedding_pairs = []
        text_provider = NetworkSingleTextProvider(
            text_terms_mapper=terms_with_embeddings_terms_mapper,
            pair_handling_func=lambda pair: term_embedding_pairs.append(pair))

        # Encoding input
        BaseInputEncoder.to_tsv(
            get_sample_filepath=lambda data_type, experiment: NetworkIOUtils.get_input_sample_filepath(
                experiment=experiment,
                data_type=data_type),
            get_opinion_filepath=lambda data_type, experiment: NetworkIOUtils.get_input_opinions_filepath(
                experiment=experiment,
                data_type=data_type),
            experiment=experiment,
            terms_per_context=terms_per_context,
            create_formatter_func=lambda data_type: NetworkInputEncoder.__create_sample_formatter(
                data_type=data_type,
                experiment=experiment,
                text_provider=text_provider),
            write_header_func=lambda _: True)

        return term_embedding_pairs

    @staticmethod
    def compose_and_save_term_embeddings_and_vocabulary(experiment, term_embedding_pairs):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(term_embedding_pairs, list))

        # Save embedding information additionally.
        term_embedding = Embedding.from_list_of_word_embedding_pairs(
            word_embedding_pairs=term_embedding_pairs)

        # Save embedding matrix
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        embedding_filepath = NetworkIOUtils.get_embedding_filepath(experiment)
        logger.info("Saving embedding [size={shape}]: {filepath}".format(
            shape=embedding_matrix.shape,
            filepath=embedding_filepath))
        np.savez(embedding_filepath, embedding_matrix)

        # Save vocabulary
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))
        vocab_filepath = NetworkIOUtils.get_vocab_filepath(experiment)
        logger.info("Saving vocabulary [size={size}]: {filepath}".format(size=len(vocab),
                                                                         filepath=vocab_filepath))
        np.savez(vocab_filepath, vocab)

    @staticmethod
    def __create_sample_formatter(data_type, experiment, text_provider):
        return NetworkSampleFormatter(
            data_type=data_type,
            label_provider=MultipleLabelProvider(label_scaler=experiment.DataIO.LabelsScaler),
            text_provider=text_provider,
            balance=False)
