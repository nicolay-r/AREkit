import collections
import logging
from collections import OrderedDict

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.experiment.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.common.experiment.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.experiment.input.repositories.sample import BaseInputSamplesRepository
from arekit.contrib.networks.core.data.serializing import NetworkSerializationData
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.helper_embedding import EmbeddingHelper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.input.providers.text import NetworkSingleTextProvider
from arekit.contrib.networks.core.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.core.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.embeddings.base import Embedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkInputHelper(object):

    # region private methods

    @staticmethod
    def __create_and_populate_repositories(
            opinions_target,
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
            columns_provider=OpinionColumnsProvider(),
            rows_provider=opinion_row_provider,
            storage=opinions_storage)

        samples_repo = BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
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

    @staticmethod
    def __perform_writing(experiment, terms_per_context, balance,
                          handle_term_embedding_pairs):
        """
        Perform experiment input serialization
        """
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))
        assert(callable(handle_term_embedding_pairs))

        term_embedding_pairs = collections.OrderedDict()

        for data_type in experiment.DocumentOperations.DataFolding.iter_supported_data_types():

            # Create annotated collection per each type.
            experiment.DataIO.Annotator.serialize_missed_collections(data_type=data_type,
                                                                     doc_ops=experiment.DocumentOperations,
                                                                     opin_ops=experiment.OpinionOperations)

            opinion_provider = OpinionProvider.create(
                read_news_func=lambda news_id: experiment.DocumentOperations.read_news(news_id),
                iter_news_opins_for_extraction=lambda news_id:
                experiment.OpinionOperations.iter_opinions_for_extraction(doc_id=news_id,
                                                                          data_type=data_type),
                parsed_news_it_func=lambda: NetworkInputHelper.__iter_parsed_news_func(
                    doc_ops=experiment.DocumentOperations,
                    data_type=data_type),
                terms_per_context=terms_per_context)

            # Composing input.
            NetworkInputHelper.__create_and_populate_repositories(
                samples_target=experiment.ExperimentIO.create_samples_writer_target(),
                opinions_target=experiment.ExperimentIO.create_opinions_writer_target(),
                opinion_provider=opinion_provider,
                sample_storage=experiment.ExperimentIO.create_samples_writer(data_type=data_type, balance=balance),
                opinions_storage=experiment.ExperimentIO.create_opinions_writer(data_type=data_type),
                exp_data=experiment.DataIO,
                entity_to_group_func=experiment.entity_to_group,
                data_type=data_type,
                term_embedding_pairs=term_embedding_pairs)

        if handle_term_embedding_pairs is not None:
            handle_term_embedding_pairs(term_embedding_pairs)

    @staticmethod
    def __iter_parsed_news_func(doc_ops, data_type):
        assert(isinstance(doc_ops, DocumentOperations))
        return doc_ops.iter_parsed_news(doc_ops.iter_news_indices(data_type))

    @staticmethod
    def __compose_and_save_term_embeddings_and_vocabulary(embedding_target, vocab_target, term_embedding_pairs):
        assert(isinstance(term_embedding_pairs, OrderedDict))

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(iter(term_embedding_pairs.items()))
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))

        # Save embedding matrix
        EmbeddingHelper.save_embedding(data=embedding_matrix, target=embedding_target)
        EmbeddingHelper.save_vocab(data=vocab, target=vocab_target)

        del embedding_matrix

    # endregion

    @staticmethod
    def prepare(experiment, terms_per_context, balance):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

        NetworkInputHelper.__perform_writing(
            experiment=experiment,
            terms_per_context=terms_per_context,
            balance=balance,
            handle_term_embedding_pairs=lambda term_embedding_pairs:
                NetworkInputHelper.__compose_and_save_term_embeddings_and_vocabulary(
                    vocab_target=experiment.ExperimentIO.get_vocab_target(),
                    embedding_target=experiment.ExperimentIO.get_embedding_target(),
                    term_embedding_pairs=term_embedding_pairs))
