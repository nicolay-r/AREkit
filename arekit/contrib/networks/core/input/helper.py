import collections
import logging

from arekit.common.data.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.opinions import InputTextOpinionProvider
from arekit.common.data.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.data.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.news.parser import NewsParser
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.input.providers.text import NetworkSingleTextProvider
from arekit.contrib.networks.core.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.core.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.embeddings.base import Embedding
from arekit.contrib.utils.pipeline import ppl_parsed_news_to_opinion_linkages, ppl_text_ids_to_parsed_news, \
    ppl_parsed_to_annotation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkInputHelper(object):

    # region private methods

    @staticmethod
    def __create_samples_repo(exp_ctx, text_provider):
        assert(isinstance(exp_ctx, NetworkSerializationContext))

        sample_row_provider = NetworkSampleRowProvider(
            label_provider=exp_ctx.LabelProvider,
            text_provider=text_provider,
            frames_connotation_provider=exp_ctx.FramesConnotationProvider,
            frame_role_label_scaler=exp_ctx.FrameRolesLabelScaler,
            pos_terms_mapper=PosTermsMapper(exp_ctx.PosTagger))

        return BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
            rows_provider=sample_row_provider,
            storage=BaseRowsStorage())

    @staticmethod
    def __create_opinions_repo():
        return BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=BaseRowsStorage())

    @staticmethod
    def __add_term_embedding(dict_data, term, emb_vector):
        if term in dict_data:
            return
        dict_data[term] = emb_vector

    @staticmethod
    def __perform_writing(exp_ctx, exp_io, doc_ops, data_type, opinion_provider,
                          terms_per_context, balance, text_provider):
        """ Perform experiment input serialization
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

        opinions_repo = NetworkInputHelper.__create_opinions_repo()

        samples_repo = NetworkInputHelper.__create_samples_repo(
            exp_ctx=exp_ctx, text_provider=text_provider)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               doc_ids=list(doc_ops.iter_doc_ids(data_type)),
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              doc_ids=list(doc_ops.iter_doc_ids(data_type)),
                              desc="sample")

        if exp_io.balance_samples(data_type=data_type, balance=balance):
            samples_repo.balance()

        # Write repositories
        samples_repo.write(writer=exp_io.create_samples_writer(),
                           target=exp_io.create_samples_writer_target(data_type=data_type))

        opinions_repo.write(writer=exp_io.create_opinions_writer(),
                            target=exp_io.create_opinions_writer_target(data_type=data_type))

    # endregion

    @staticmethod
    def prepare(exp_ctx, exp_io, doc_ops, opin_ops, terms_per_context, balance, value_to_group_id_func, text_parser):
        assert(isinstance(exp_ctx, NetworkSerializationContext))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

        term_embedding_pairs = collections.OrderedDict()

        text_terms_mapper = StringWithEmbeddingNetworkTermMapping(
            predefined_embedding=exp_ctx.WordEmbedding,
            string_entities_formatter=exp_ctx.StringEntityFormatter,
            string_emb_entity_formatter=exp_ctx.StringEntityEmbeddingFormatter)

        text_provider = NetworkSingleTextProvider(
            text_terms_mapper=text_terms_mapper,
            pair_handling_func=lambda pair: NetworkInputHelper.__add_term_embedding(
                dict_data=term_embedding_pairs,
                term=pair[0],
                emb_vector=pair[1]))

        for data_type in exp_ctx.DataFolding.iter_supported_data_types():

            pipeline = BasePipeline(
                ppl_text_ids_to_parsed_news(
                    parse_news_func=lambda doc_id: NewsParser.parse(
                        news=doc_ops.get_doc(doc_id),
                        text_parser=text_parser))
                +
                ppl_parsed_to_annotation(annotator=exp_ctx.Annotator,
                                         data_type=data_type,
                                         opin_ops=opin_ops)
                +
                ppl_parsed_news_to_opinion_linkages(value_to_group_id_func=value_to_group_id_func,
                                                    terms_per_context=terms_per_context)
            )

            NetworkInputHelper.__perform_writing(
                exp_ctx=exp_ctx,
                exp_io=exp_io,
                doc_ops=doc_ops,
                data_type=data_type,
                opinion_provider=InputTextOpinionProvider(pipeline),
                terms_per_context=terms_per_context,
                balance=balance,
                text_provider=text_provider)

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(iter(term_embedding_pairs.items()))
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))

        # Save embedding matrix
        exp_io.save_embedding(data=embedding_matrix)
        exp_io.save_vocab(data=vocab)

        del embedding_matrix
