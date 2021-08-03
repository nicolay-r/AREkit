from arekit.common.experiment.extract.opinions import iter_linked_text_opins
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations


class OpinionProvider(object):
    """
    TextOpinion iterator
    """

    def __init__(self, linked_text_opins_it_func):
        assert(callable(linked_text_opins_it_func))
        self.__linked_text_opins_it_func = linked_text_opins_it_func

    @classmethod
    def from_experiment(cls, doc_ops, opin_ops, data_type, parsed_news_it_func, terms_per_context):
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(terms_per_context, int))
        assert(callable(parsed_news_it_func))

        def it_func():
            return iter_linked_text_opins(doc_ops=doc_ops,
                                          opin_ops=opin_ops,
                                          data_type=data_type,
                                          terms_per_context=terms_per_context,
                                          parsed_news_it=parsed_news_it_func())

        return cls(linked_text_opins_it_func=it_func)

    def iter_linked_opinion_wrappers(self):
        for data in self.__linked_text_opins_it_func():
            yield data
