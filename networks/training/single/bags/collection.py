import numpy as np

from arekit.common.linked.text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.networks.training.single.bags.bag import Bag
from arekit.contrib.networks.sample import InputSample


class BagsCollection:

    def __init__(self, bags):
        assert(isinstance(bags, list))
        self.__bags = bags

    # region classmethods

    @classmethod
    def from_linked_text_opinions(
            cls,
            text_opinion_collection,
            data_type,
            bag_size,
            create_sample_func,
            create_empty_sample_func,
            shuffle):
        assert(isinstance(bag_size, int) and bag_size > 0)
        assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))
        assert(callable(create_sample_func))
        assert(isinstance(shuffle, bool))

        bags = []

        for linked_wrap in text_opinion_collection.iter_wrapped_linked_text_opinions():
            bags.append(Bag(linked_wrap.First.Sentiment))
            for opinion in linked_wrap:
                assert(isinstance(opinion, TextOpinion))

                if len(bags[-1]) == bag_size:
                    bags.append(Bag(opinion.Sentiment))

                s = create_sample_func(opinion)
                assert(isinstance(s, InputSample))

                bags[-1].add_sample(s)

            if len(bags[-1]) == 0:
                bags = bags[:-1]
                continue

            cls.__complete_last_bag(bags, bag_size)

        if shuffle:
            np.random.shuffle(bags)

        return cls(bags)

    # endregion

    # region private methods

    @staticmethod
    def __complete_last_bag(bags, bag_size):
        last_bag = bags[-1]
        while len(last_bag) < bag_size:
            last_bag.add_sample(last_bag.Samples[-1])

    @staticmethod
    def __is_bag_contains_text_opinion_from_set(bags_list, text_opinion_ids_set):
        """
        Check the presence of at least single sample in text_opinion_ids_set
        text_opinion_ids_set: set or None
            set of opinion ids.
            None -- keep all bags.
        """

        if text_opinion_ids_set is None:
            return True

        for bag in bags_list:
            for sample in bag:
                if sample.TextOpinionID in text_opinion_ids_set:
                    return True

        return False

    # endregion

    # region public methods

    def iter_by_groups(self, bags_per_group, text_opinion_ids_set=None):
        """
        text_opinion_ids_set: set or None
            set of opinion ids.
            None -- keep all bags.
        """
        assert(type(bags_per_group) == int and bags_per_group > 0)
        assert(isinstance(text_opinion_ids_set, set) or text_opinion_ids_set is None)

        def __check(bags_list):
            return self.__is_bag_contains_text_opinion_from_set(
                bags_list=bags_list,
                text_opinion_ids_set=text_opinion_ids_set)

        groups_count = len(self.__bags) / bags_per_group
        end = 0
        for index in xrange(groups_count):

            begin = index * bags_per_group
            end = begin + bags_per_group
            bags = self.__bags[begin:end]

            if __check(bags):
                yield bags

        last_group = []
        while len(last_group) != bags_per_group:
            if not end < len(self.__bags):
                end = 0
            last_group.append(self.__bags[end])
            end += 1

        if __check(last_group):
            yield last_group

    # endregion

    # region overriden methods

    def __iter__(self):
        for bag in self.__bags:
            yield bag

    def __len__(self):
        return len(self.__bags)

    # endregion
