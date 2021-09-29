import collections
import logging
from collections import OrderedDict

from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.experiment.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.common.experiment.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.experiment.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.experiment.input.storages.base import BaseRowsStorage
from arekit.contrib.networks.core.input.data_serialization import NetworkSerializationData
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
    def __create_samples_repo(exp_data, term_embedding_pairs, entity_to_group_func, sample_writer):
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

        return BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
            rows_provider=sample_row_provider,
            storage=BaseRowsStorage(),
            writer=sample_writer)

    @staticmethod
    def __create_opinions_repo(opinions_writer, data_type):
        assert(isinstance(data_type, DataType))
        return BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=BaseRowsStorage(),
            writer=opinions_writer)

    @staticmethod
    def __add_term_embedding(dict_data, term, emb_vector):
        if term in dict_data:
            return
        dict_data[term] = emb_vector

    @staticmethod
    def __perform_writing(experiment, data_type, terms_per_context, balance, term_embedding_pairs):
        """
        Perform experiment input serialization
        """
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

        doc_ops = experiment.DocumentOperations
        opin_ops = experiment.OpinionOperations

        # TODO. 208. This should be done in advance.
        experiment.DataIO.Annotator.serialize_missed_collections(
            data_type=data_type,
            doc_ops=doc_ops,
            opin_ops=opin_ops)

        opinion_provider = OpinionProvider.create(
            read_news_func=lambda news_id: doc_ops.read_news(news_id),
            iter_news_opins_for_extraction=lambda news_id:
                # TODO. 208. Annotated opinions should be passed here.
                opin_ops.iter_opinions_for_extraction(doc_id=news_id,
                                                      data_type=data_type),
            parsed_news_it_func=lambda: doc_ops.iter_parsed_news(
                doc_ops.iter_doc_ids(data_type)),
            terms_per_context=terms_per_context)

        # Composing input.
        opinions_repo = NetworkInputHelper.__create_opinions_repo(
            opinions_writer=experiment.ExperimentIO.create_opinions_writer(data_type=data_type),
            data_type=data_type)

        samples_repo = NetworkInputHelper.__create_samples_repo(
            sample_writer=experiment.ExperimentIO.create_samples_writer(data_type=data_type, balance=balance),
            exp_data=experiment.DataIO,
            entity_to_group_func=experiment.entity_to_group,
            term_embedding_pairs=term_embedding_pairs)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               target=experiment.ExperimentIO.create_opinions_writer_target(data_type=data_type),
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              target=experiment.ExperimentIO.create_samples_writer_target(data_type=data_type),
                              desc="sample")

    # endregion

    @staticmethod
    def prepare(experiment, terms_per_context, balance):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

        term_embedding_pairs = collections.OrderedDict()

        for data_type in experiment.DocumentOperations.DataFolding.iter_supported_data_types():
            NetworkInputHelper.__perform_writing(# TODO. Remove this experiment instance.
                                                 experiment=experiment,
                                                 data_type=data_type,
                                                 terms_per_context=terms_per_context,
                                                 balance=balance,
                                                 term_embedding_pairs=term_embedding_pairs)

        # Assign targets
        vocab_target = experiment.ExperimentIO.get_vocab_target()
        embedding_target = experiment.ExperimentIO.get_embedding_target()

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(iter(term_embedding_pairs.items()))
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))

        # Save embedding matrix
        EmbeddingHelper.save_embedding(data=embedding_matrix, target=embedding_target)
        EmbeddingHelper.save_vocab(data=vocab, target=vocab_target)

        del embedding_matrix
