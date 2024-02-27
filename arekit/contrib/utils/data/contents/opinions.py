from arekit.common.data.input.providers.const import IDLE_MODE
from arekit.common.data.input.providers.contents import ContentsProvider
from arekit.common.linkage.base import LinkedDataWrapper
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text_opinions.base import TextOpinion


class InputTextOpinionProvider(ContentsProvider):

    def __init__(self, pipeline):
        """ NOTE: it is important that the output of the pipeline
            results in a TextOpinionLinkage instances.
            pipeline: id -> ... -> TextOpinionLinkage[]
        """
        assert(isinstance(pipeline, list))
        self.__pipeline = pipeline
        self.__current_id = None

    # endregion

    def __assign_ids(self, linkage):
        """ Perform IDs assignation.
        """
        assert(isinstance(linkage, TextOpinionsLinkage))
        for text_opinion in linkage:
            assert(isinstance(text_opinion, TextOpinion))
            text_opinion.set_text_opinion_id(self.__current_id)
            self.__current_id += 1

    def from_doc_ids(self, doc_ids, idle_mode=False):
        self.__current_id = 0

        ctx = PipelineContext(d={
            "result": doc_ids,
            IDLE_MODE: idle_mode
        })

        # Launching pipeline with the passed context
        BasePipelineLauncher.run(pipeline=self.__pipeline, pipeline_ctx=ctx)

        for linkage in ctx.provide("result"):
            assert(isinstance(linkage, LinkedDataWrapper))
            if isinstance(linkage, TextOpinionsLinkage):
                self.__assign_ids(linkage)
            yield linkage
