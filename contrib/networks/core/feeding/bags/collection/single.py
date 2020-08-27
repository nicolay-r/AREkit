from arekit.contrib.networks.core.feeding.bags.bag import Bag
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.sample import InputSample
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow


class SingleBagsCollection(BagsCollection):

    @staticmethod
    def __complete_last_bag(bags, bag_size):
        last_bag = bags[-1]
        while len(last_bag) < bag_size:
            last_bag.add_sample(last_bag.Samples[-1])

    @classmethod
    def _fill_bags_list_with_linked_text_opinions(cls, bags, parsed_rows, bag_size, create_sample_func,
                                                  create_empty_sample_func):
        assert(isinstance(parsed_rows, list))
        assert(len(parsed_rows) > 0)
        assert(isinstance(bags, list))
        assert(callable(create_sample_func))

        bags.append(Bag(parsed_rows[0].Sentiment))
        for parsed_row in parsed_rows:
            assert(isinstance(parsed_row, ParsedSampleRow))

            if len(bags[-1]) == bag_size:
                bags.append(Bag(parsed_row.Sentiment))

            s = create_sample_func(parsed_row)
            assert(isinstance(s, InputSample))

            bags[-1].add_sample(s)

        if len(bags[-1]) == 0:
            del bags[-1]
            return

        SingleBagsCollection.__complete_last_bag(bags=bags, bag_size=bag_size)
