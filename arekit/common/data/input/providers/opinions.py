from arekit.common.data.input.pipeline import text_opinions_iter_pipeline
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext


class InputTextOpinionProvider(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, BasePipeline))
        self.__pipeline = pipeline

    # endregion

    @classmethod
    def create(cls, iter_doc_opins, value_to_group_id_func,
               parse_news_func, terms_per_context):

        pipeline = text_opinions_iter_pipeline(
            parse_news_func=parse_news_func,
            value_to_group_id_func=value_to_group_id_func,
            iter_doc_opins=iter_doc_opins,
            terms_per_context=terms_per_context)

        return cls(pipeline)

    def iter_linked_opinions(self, doc_ids):
        ctx = PipelineContext({"src": doc_ids})
        self.__pipeline.run(ctx)
        for linkage in ctx.provide("src"):
            assert(isinstance(linkage, TextOpinionsLinkage))
            yield linkage
