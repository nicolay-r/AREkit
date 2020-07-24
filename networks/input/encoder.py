import logging

import numpy as np
from os.path import join

from arekit.common.embeddings.base import Embedding
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.entities.str_emb_fmt import StringWordEmbeddingEntityFormatter
from arekit.networks.input.embedding.matrix import create_term_embedding_matrix
from arekit.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.networks.input.formatters.sample import NetworkSampleFormatter
from arekit.networks.input.providers.text.single import NetworkSingleTextProvider
from arekit.networks.input.terms_mapping import StringWithEmbeddingNetworkTermMapping


logger = logging.getLogger(__name__)


class NetworkInputEncoder(object):

    TERM_EMBEDDING_FILENAME = 'term_embedding.gz'
    VOCABULARY_FILENAME = "vocab.txt.gz"

    @staticmethod
    def to_tsv_with_embedding_and_vocabulary(experiment, config):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))

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

        target_dir = NetworkInputEncoder.get_samples_dir(experiment)

        # Encoding input
        BaseInputEncoder.to_tsv(
            balance=False,
            # TODO. Move this to the method.
            # TODO. Access this method from data_handler.
            out_dir=target_dir,
            experiment=experiment,
            create_formatter_func=lambda data_type:
                NetworkInputEncoder.__create_sample_formatter(data_type=data_type,
                                                              experiment=experiment,
                                                              text_provider=text_provider))

        # Save embedding information additionally.
        term_embedding = Embedding.from_list_of_word_embedding_pairs(
            word_embedding_pairs=term_embedding_pairs)

        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        config.set_term_embedding(embedding_matrix)
        embedding_filepath = join(target_dir, NetworkInputEncoder.TERM_EMBEDDING_FILENAME)
        logger.info("Saving embedding: {}".format(embedding_filepath))
        np.savez(embedding_filepath, embedding_matrix)

        vocab = TermsEmbeddingOffsets.iter_words_vocabulary(words_embedding=term_embedding)
        vocab_filepath = join(target_dir, NetworkInputEncoder.VOCABULARY_FILENAME)
        logger.info("Saving vocabulary: {}".format(vocab_filepath))
        np.savez(vocab_filepath, list(vocab))

    @staticmethod
    def check_files_existance(target_dir, data_type):
        assert(isinstance(target_dir, unicode))
        assert(isinstance(data_type, DataType))
        # TODO. Update
        return False

    @staticmethod
    def __create_sample_formatter(data_type, experiment, text_provider):
        return NetworkSampleFormatter(
            data_type=data_type,
            label_provider=MultipleLabelProvider(label_scaler=experiment.DataIO.LabelsScaler),
            text_provider=text_provider)

    @staticmethod
    def get_samples_dir(experiment):
        return experiment.get_input_samples_dir()

