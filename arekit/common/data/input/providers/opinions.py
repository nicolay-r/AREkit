from arekit.common.experiment.pipelines.text_opinoins_input import process_input_text_opinions
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext


class OpinionProvider(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, BasePipeline))
        self.__pipeline = pipeline

    # endregion

    @classmethod
    def create(cls, iter_doc_opins, value_to_group_id_func,
               parse_news_func, terms_per_context):
        assert(callable(iter_doc_opins))
        assert(callable(value_to_group_id_func))
        assert(isinstance(terms_per_context, int))
        assert(callable(parse_news_func))

        pipeline = process_input_text_opinions(
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
            print(linkage.Tag)
            yield linkage
