import numpy as np

from arekit.common.embeddings.base import Embedding
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.entities.str_emb_fmt import StringWordEmbeddingEntityFormatter
from arekit.networks.input.embedding.matrix import create_term_embedding_matrix
from arekit.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.networks.input.formatters.sample import NetworkSampleFormatter
from arekit.networks.input.providers.text.single import SingleTextProvider
from arekit.networks.input.terms_mapping import EmbeddedTermMapping


class NetworkInputEncoder(object):

    TERM_EMBEDDING_FILENAME = 'term_embedding.gz'
    VOCABULARY_FILENAME = "vocab.txt.gz"

    @staticmethod
    def to_tsv_with_embedding_and_vocabulary(experiment, config):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))

        predefined_embedding = experiment.DataIO.WordEmbedding
        predefined_embedding.set_stemmer(experiment.DataIO.Stemmer)

        text_terms_mapper = EmbeddedTermMapping(
            predefined_embedding=predefined_embedding,
            string_entities_formatter=experiment.DataIO.StringEntityFormatter,
            string_emb_entity_formatter=StringWordEmbeddingEntityFormatter())

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type)

        term_embedding_pairs = []
        text_provider = SingleTextProvider(
            text_terms_mapper=text_terms_mapper,
            write_embedding_pair_func=lambda pair: term_embedding_pairs.append(pair))

        # Encoding input
        BaseInputEncoder.to_tsv(
            experiment=experiment,
            create_formatter_func=lambda data_type:
                NetworkSampleFormatter(data_type=data_type,
                                       label_provider=MultipleLabelProvider(label_scaler=experiment.DataIO.LabelsScaler),
                                       text_provider=text_provider))

        # Save embedding information additionally.
        term_embedding = Embedding.from_list_of_word_embedding_pairs(
            word_embedding_pairs=term_embedding_pairs)

        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        config.set_term_embedding(embedding_matrix)
        np.savez(NetworkInputEncoder.TERM_EMBEDDING_FILENAME, embedding_matrix)

        vocab = TermsEmbeddingOffsets.iter_words_vocabulary(words_embedding=term_embedding)
        np.savez(NetworkInputEncoder.VOCABULARY_FILENAME, list(vocab))
