from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text_opinions.base import TextOpinion


class InputTextOpinionProvider(object):

    def __init__(self, pipeline):
        """ NOTE: it is important that the output of the pipeline
            results in a TextOpinionLinkage instances.
        """
        assert(isinstance(pipeline, BasePipeline))
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

    def iter_linked_opinions(self, doc_ids):
        self.__current_id = 0
        for linkage in self.__pipeline.run(doc_ids):
            assert(isinstance(linkage, TextOpinionsLinkage))
            self.__assign_ids(linkage)
            yield linkage
