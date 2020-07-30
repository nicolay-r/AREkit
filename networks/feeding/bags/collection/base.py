import numpy as np
from tqdm import tqdm

from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.networks.input.rows_parser import ParsedSampleRow


class BagsCollection(object):

    def __init__(self, bags):
        assert(isinstance(bags, list))
        self._bags = bags

    @classmethod
    def _fill_bags_list_with_linked_text_opinions(cls, bags, parsed_rows, bag_size, create_sample_func,
                                                  create_empty_sample_func):
        raise NotImplementedError()

    @classmethod
    def from_formatted_samples(cls,
                               samples_reader,
                               bag_size,
                               label_scaler,
                               create_sample_func,
                               create_empty_sample_func,
                               shuffle,
                               desc=None):
        assert(isinstance(samples_reader, InputSampleReader))
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(bag_size, int) and bag_size > 0)
        assert(callable(create_sample_func))
        assert(callable(create_empty_sample_func))
        assert(isinstance(shuffle, bool))

        bags = []

        linked_rows_iter = tqdm(iterable=samples_reader.iter_rows_linked_by_text_opinions(),
                                desc=desc)

        for linked_rows in linked_rows_iter:
            assert(len(linked_rows) > 0)

            cls._fill_bags_list_with_linked_text_opinions(
                bags=bags,
                parsed_rows=[ParsedSampleRow.parse(row=row, labels_scaler=label_scaler)
                             for row in linked_rows],
                bag_size=bag_size,
                create_sample_func=create_sample_func,
                create_empty_sample_func=create_empty_sample_func)

        if shuffle:
            np.random.shuffle(bags)

        return cls(bags)

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
                if sample.ID in text_opinion_ids_set:
                    return True

        return False

    def get_groups_count(self, bags_per_group):
        return len(self._bags) / bags_per_group

    def shuffle(self):
        np.random.shuffle(self._bags)

    def iter_by_groups(self, bags_per_group, text_opinion_ids_set=None):
        """
        text_opinion_ids_set: set or None
            set of opinion ids.
            None -- keep all bags.
        """
        assert(type(bags_per_group) == int and bags_per_group > 0)
        assert(isinstance(text_opinion_ids_set, set) or text_opinion_ids_set is None)
        assert(len(self._bags) > 0)

        def __check(bags_list):
            return self.__is_bag_contains_text_opinion_from_set(
                bags_list=bags_list,
                text_opinion_ids_set=text_opinion_ids_set)

        end = 0
        for index in xrange(self.get_groups_count(bags_per_group)):

            begin = index * bags_per_group
            end = begin + bags_per_group
            bags = self._bags[begin:end]

            if __check(bags):
                yield bags

        last_group = []
        while len(last_group) != bags_per_group:
            if not end < len(self._bags):
                end = 0
            last_group.append(self._bags[end])
            end += 1

        if __check(last_group):
            yield last_group

    def __iter__(self):
        for bag in self._bags:
            yield bag

    def __len__(self):
        return len(self._bags)
