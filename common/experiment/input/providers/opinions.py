from arekit.common.experiment.extract.opinions import iter_linked_text_opins
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.labels.base import Label


class OpinionProvider(object):
    """
    TextOpinion iterator + balancing.
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

    # region private methods

    @staticmethod
    def __get_label_by_linked_wrap(linked_wrap):
        return linked_wrap.get_linked_label()

    def __iter(self, linked_wrap_check_func=None, after_every_it_cb=None):
        assert(callable(linked_wrap_check_func) or linked_wrap_check_func is None)
        assert(callable(after_every_it_cb) or after_every_it_cb is None)

        for parsed_news, linked_wrap in self.__linked_text_opins_it_func():

            if linked_wrap_check_func is not None and not linked_wrap_check_func(linked_wrap):
                continue

            yield parsed_news, linked_wrap

    def __iter_filtered_by_same_label_and_limited(self, label, count):

        it_data = self.__iter(
            linked_wrap_check_func=lambda linked_wrap: self.__get_label_by_linked_wrap(linked_wrap) == label)

        for data in it_data:
            if count == 0:
                break
            yield data
            count -= 1

    def __iter_balanced(self, supported_labels):
        assert(isinstance(supported_labels, list))

        counts = {}
        for label in supported_labels:
            assert(isinstance(label, Label))
            counts[label] = 0

        for parsed_news, linked_wrap in self.__iter():
            counts[self.__get_label_by_linked_wrap(linked_wrap)] += 1
            yield parsed_news, linked_wrap

        top = max(counts.values())

        for label in counts.iterkeys():

            if counts[label] == 0:
                continue

            left = top - counts[label]
            for pair in self.__iter_filtered_by_same_label_and_limited(label=label, count=left):
                yield pair

    # endregion

    def iter_linked_opinion_wrappers(self, balance, supported_labels):
        assert(isinstance(balance, bool))
        assert(isinstance(supported_labels, list) or supported_labels is None)

        return self.__iter() if not balance \
            else self.__iter_balanced(supported_labels=supported_labels)
