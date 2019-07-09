import numpy as np

from core.common.text_opinions.text_opinion import TextOpinion
from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.networks.context.training.bags.bag import Bag

from core.networks.context.sample import InputSample


class BagsCollection:

    def __init__(self, bags):
        assert(isinstance(bags, list))
        self.__bags = bags

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

        for text_opinions in text_opinion_collection.iter_by_linked_text_opinions():
            bags.append(Bag(text_opinions[0].Sentiment))
            for opinion in text_opinions:
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

    @staticmethod
    def __complete_last_bag(bags, bag_size):
        last_bag = bags[-1]
        while len(last_bag) < bag_size:
            last_bag.add_sample(last_bag.Samples[-1])

    def iter_by_groups(self, bags_per_group):
        assert(type(bags_per_group) == int and bags_per_group > 0)

        groups_count = len(self.__bags) / bags_per_group
        end = 0
        for index in range(groups_count):
            begin = index * bags_per_group
            end = begin + bags_per_group
            yield self.__bags[begin:end]

        last_group = []
        while len(last_group) != bags_per_group:
            if not end < len(self.__bags):
                end = 0
            last_group.append(self.__bags[end])
            end += 1
        yield last_group

    def __iter__(self):
        for bag in self.__bags:
            yield bag

    def __len__(self):
        return len(self.__bags)
