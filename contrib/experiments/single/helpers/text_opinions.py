from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.base import Opinion
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.text_opinions.base import TextOpinion
from arekit.networks.labeling.base import LabelsHelper


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

    def iter_opinion_collections(self, create_collection_func, label_calculation_mode):
        assert(callable(create_collection_func))
        assert(isinstance(label_calculation_mode, unicode))

        for news_id in self.__collection.iter_unique_news_ids():
            collection = self.__to_opinion_collection(create_collection_func=create_collection_func,
                                                      news_id=news_id,
                                                      label_mode=label_calculation_mode)

            yield collection, news_id

    def debug_labels_statistic(self):
        norm, _ = self.get_statistic()
        total = len(self.__collection)
        print "Extracted relation collection: {}".format(self.__name)
        print "\tTotal: {}".format(total)
        for i, value in enumerate(norm):
            label = self.__labels_helper.create_label_from_uint(i)
            print "\t{}: {:.2f}%".format(label.to_str(), value)

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

    def __to_opinion_collection(self, create_collection_func, news_id, label_mode):
        assert(callable(create_collection_func))
        assert(isinstance(news_id, int))
        assert(isinstance(label_mode, unicode))

        collection = create_collection_func()

        for liked_text_opinions in self.__collection.iter_by_linked_text_opinions():

            first = liked_text_opinions[0]
            assert(isinstance(first, TextOpinion))

            # TODO. Use textOpinionHelper.
            if first.NewsID != news_id:
                continue

            label = self.__labels_helper.create_label_from_text_opinions(
                text_opinion_labels=[text_opinion.Sentiment for text_opinion in liked_text_opinions],
                label_creation_mode=label_mode)

            opinion_list = self.__labels_helper.create_opinions_from_text_opinion_and_label(
                text_opinion=first,
                label=label)

            for opinion in opinion_list:
                self.__opitional_add_opinion(opinion=opinion,
                                             collection=collection)

        return collection

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
        for group in self.__collection.iter_by_linked_text_opinions():
            key = len(group)
            if key not in statistic:
                statistic[key] = 1
            else:
                statistic[key] += 1
        return statistic

    # endregion

