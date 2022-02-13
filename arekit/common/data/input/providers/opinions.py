from arekit.common.data.input.pipeline import text_opinions_iter_pipeline
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text_opinions.base import TextOpinion


class InputTextOpinionProvider(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, BasePipeline))
        self.__pipeline = pipeline
        self.__current_id = None

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

    def __assign_ids(self, linkage):
        """ Perform IDs assignation.
        """
        assert(isinstance(linkage, TextOpinionsLinkage))
        for text_opinion in linkage:
            assert(isinstance(text_opinion, TextOpinion))
            text_opinion.set_text_opinion_id(self.__current_id)
            self.__current_id += 1

    def iter_linked_opinions(self, doc_ids):
        ctx = PipelineContext({"src": doc_ids})
        self.__pipeline.run(ctx)
        self.__current_id = 0
        for linkage in ctx.provide("src"):
            assert(isinstance(linkage, TextOpinionsLinkage))
            self.__assign_ids(linkage)
            yield linkage
