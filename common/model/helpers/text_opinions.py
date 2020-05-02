from arekit.common.labels.base import NeutralLabel
from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.text_opinions.text_opinion import TextOpinion


class LabeledLinkedTextOpinionCollectionHelper:

    def __init__(self, collection, labels_helper, name):
        assert(isinstance(collection, LabeledLinkedTextOpinionCollection))
        assert(isinstance(labels_helper, LabelsHelper))
        assert(isinstance(name, unicode))
        self.__collection = collection
        self.__labels_helper = labels_helper
        self.__name = name

    # region properties

    @property
    def Name(self):
        return self.__name

    # endregion

    # region public methods

    def iter_converted_to_opinion_collections(self, create_collection_func, label_calc_mode):
        assert(callable(create_collection_func))
        assert(isinstance(label_calc_mode, unicode))

        for news_id in self.__collection.iter_unique_news_ids():
            collection = create_collection_func()
            for doc_opinion in self.__iter_opinions(news_id=news_id, label_mode=label_calc_mode):
                self.__opitional_add_opinion(opinion=doc_opinion,
                                             collection=collection)

            yield collection, news_id

    def debug_labels_statistic(self):
        norm, stat = self.get_statistic()
        total = len(self.__collection)
        print "Extracted relation collection: {}".format(self.__name)
        print "\tTotal: {}".format(total)
        for i, value in enumerate(norm):
            label = self.__labels_helper.create_label_from_uint(i)
            print "\t{}: {:.2f}%\t({} relations)".format(label.to_str(), value, stat[i])

    def debug_unique_relations_statistic(self):
        statistic = self.__get_group_statistic()
        total = sum(list(statistic.itervalues()))
        print "Unique linked_text_opinions statistic: {}".format(self.__name)
        print "\tTotal: {}".format(total)
        for key, value in sorted(statistic.iteritems()):
            print "\t{} -- {} ({:.2f}%)".format(key, value, 100.0 * value / total)
            total += value

    def get_statistic(self):
        stat = [0] * self.__labels_helper.get_classes_count()
        for text_opinion in self.__collection:
            assert(isinstance(text_opinion, TextOpinion))
            stat[text_opinion.Sentiment.to_uint()] += 1

        total = sum(stat)
        norm = [100.0 * value / total if total > 0 else 0 for value in stat]
        return norm, stat

    # endregion

    # region private methods

    def __iter_opinions(self, news_id, label_mode):
        assert(isinstance(news_id, int))
        assert(isinstance(label_mode, unicode))

        for linked_wrap in self.__collection.iter_wrapped_linked_text_opinions():
            assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))

            if linked_wrap.RelatedNewsID != news_id:
                continue

            label = self.__labels_helper.create_label_from_text_opinions(
                text_opinion_labels=linked_wrap,
                label_creation_mode=label_mode)

            opinions_it = self.__labels_helper.iter_opinions_from_text_opinion_and_label(
                text_opinion=linked_wrap.FirstOpinion,
                label=label)

            for opinion in opinions_it:
                yield opinion

    @staticmethod
    def __opitional_add_opinion(opinion, collection):
        assert(isinstance(opinion, Opinion))

        if isinstance(opinion.Sentiment, NeutralLabel):
            return

        if collection.has_synonymous_opinion(opinion):
            return

        collection.add_opinion(opinion)

    def __get_group_statistic(self):
        statistic = {}
        for linked_wrap in self.__collection.iter_wrapped_linked_text_opinions():
            key = len(linked_wrap)
            if key not in statistic:
                statistic[key] = 1
            else:
                statistic[key] += 1
        return statistic

    # endregion

