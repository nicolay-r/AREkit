import logging
import os

import numpy as np
from os.path import join

from arekit.common.embeddings.base import Embedding
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.contrib.networks.entities.str_emb_fmt import StringWordEmbeddingEntityFormatter
from arekit.networks.input.embedding.matrix import create_term_embedding_matrix
from arekit.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.networks.input.formatters.sample import NetworkSampleFormatter
from arekit.networks.input.providers.text.single import NetworkSingleTextProvider
from arekit.networks.input.terms_mapping import StringWithEmbeddingNetworkTermMapping


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkInputEncoder(object):

    TERM_EMBEDDING_FILENAME = u'term_embedding'
    VOCABULARY_FILENAME = u"vocab.txt"

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

        target_dir = NetworkInputEncoder.get_samples_dir(experiment)

        # Encoding input
        BaseInputEncoder.to_tsv(balance=False,
                                out_dir=target_dir,
                                experiment=experiment,
                                terms_per_context=terms_per_context,
                                create_formatter_func=lambda data_type:
                                    NetworkInputEncoder.__create_sample_formatter(data_type=data_type,
                                                                                  experiment=experiment,
                                                                                  text_provider=text_provider),
                                write_header_func=lambda _: True)

        return term_embedding_pairs

    @staticmethod
    def compose_and_save_term_embeddings_and_vocabulary(target_dir, term_embedding_pairs):
        assert(isinstance(target_dir, unicode))
        assert(isinstance(term_embedding_pairs, list))

        # Save embedding information additionally.
        term_embedding = Embedding.from_list_of_word_embedding_pairs(
            word_embedding_pairs=term_embedding_pairs)

        # Save embedding matrix
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        embedding_filepath = NetworkInputEncoder.get_embedding_filepath(target_dir)
        logger.info("Saving embedding [size={shape}: {filepath}".format(
            shape=embedding_matrix.shape,
            filepath=embedding_filepath))
        np.savez(embedding_filepath, embedding_matrix)

        # Save vocabulary
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))
        vocab_filepath = NetworkInputEncoder.get_vocab_filepath(target_dir)
        logger.info("Saving vocabulary [size={size}]: {filepath}".format(size=len(vocab),
                                                                         filepath=vocab_filepath))
        np.savez(vocab_filepath, vocab)

    @staticmethod
    def check_files_existance(target_dir, data_type, experiment):
        assert(isinstance(target_dir, unicode))
        assert(isinstance(data_type, DataType))
        assert(isinstance(experiment, BaseExperiment))

        filepaths = list(BaseInputEncoder.get_filepaths(out_dir=target_dir,
                                                        data_type=data_type,
                                                        experiment=experiment))

        filepaths.extend([NetworkInputEncoder.get_vocab_filepath(target_dir),
                          NetworkInputEncoder.get_embedding_filepath(target_dir)])

        result = True
        for filepath in filepaths:
            existed = os.path.exists(filepath)
            logger.info("Check existance [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
            if not existed:
                result = False

        return result

    @staticmethod
    def get_vocab_filepath(target_dir):
        return join(target_dir, NetworkInputEncoder.VOCABULARY_FILENAME + u'.npz')

    @staticmethod
    def get_embedding_filepath(target_dir):
        return join(target_dir, NetworkInputEncoder.TERM_EMBEDDING_FILENAME + u'.npz')

    @staticmethod
    def __create_sample_formatter(data_type, experiment, text_provider):
        return NetworkSampleFormatter(
            data_type=data_type,
            label_provider=MultipleLabelProvider(label_scaler=experiment.DataIO.LabelsScaler),
            text_provider=text_provider)

    @staticmethod
    def get_samples_dir(experiment):
        return experiment.get_input_samples_dir()

