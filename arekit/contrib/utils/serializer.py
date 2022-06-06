import logging

from arekit.common.data.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.opinions import InputTextOpinionProvider
from arekit.common.data.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.data.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.news.parser import NewsParser
from arekit.common.pipeline.base import BasePipeline

from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.utils.pipelines.opinion_annotation import ppl_text_ids_to_parsed_news, ppl_parsed_to_annotation, \
    ppl_parsed_news_to_opinion_linkages

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InputDataSerializationHelper(object):

    @staticmethod
    def serialize(exp_ctx, exp_io, doc_ops, opin_ops, terms_per_context, balance,
                  data_type, value_to_group_id_func, text_parser, sample_rows_provider):
        assert(isinstance(exp_ctx, NetworkSerializationContext))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

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

        opinions_repo = BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=BaseRowsStorage())

        samples_repo = BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
            rows_provider=sample_rows_provider,
            storage=BaseRowsStorage())

        opinion_provider = InputTextOpinionProvider(pipeline)

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
