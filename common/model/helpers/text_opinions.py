from arekit.common.experiment.opinions import compose_opinion_collection
from arekit.common.linked.text_opinions.collection import LinkedTextOpinionCollection
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.text_opinion import TextOpinion


class LabeledLinkedTextOpinionCollectionHelper:

    def __init__(self, collection, labels_helper, text_opinion_helper, name):
        assert(isinstance(collection, LinkedTextOpinionCollection))
        assert(isinstance(labels_helper, LabelsHelper))
        assert(isinstance(text_opinion_helper, TextOpinionHelper))
        assert(isinstance(name, unicode))
        self.__collection = collection
        self.__labels_helper = labels_helper
        self.__text_opinion_helper = text_opinion_helper
        self.__name = name

    # region public methods

    def iter_converted_to_opinion_collections(self, create_collection_func, label_calc_mode):
        assert(callable(create_collection_func))
        assert(isinstance(label_calc_mode, unicode))

        for news_id in self.__collection.get_unique_news_ids():

            collection = compose_opinion_collection(
                create_collection_func=create_collection_func,
                linked_data_iter=self.__collection.iter_wrapped_linked_text_opinions(news_id=news_id),
                labels_helper=self.__labels_helper,
                to_opinion_func=self.__text_opinion_helper.to_opinion,
                label_calc_mode=label_calc_mode)

            yield collection, news_id

    def debug_labels_statistic(self):
        norm, stat = self.get_statistic()
        total = len(self.__collection)
        print "Extracted relation collection: {}".format(self.__name)
        print "\tTotal: {}".format(total)
        for i, value in enumerate(norm):
            label = self.__labels_helper.label_from_uint(i)
            print "\t{}: {:.2f}%\t({} relations)".format(label.to_class_str(), value, stat[i])

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
            stat[self.__labels_helper.label_to_uint(text_opinion.Sentiment)] += 1

        total = sum(stat)
        norm = [100.0 * value / total if total > 0 else 0 for value in stat]
        return norm, stat

    # endregion

    # region private methods

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

