from arekit.common.experiment.input.sample import InputSampleBase
from arekit.contrib.networks.core.feeding.bags.bag import Bag
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow


class MultiInstanceBagsCollection(BagsCollection):

    # region private methods

    @staticmethod
    def __last_bag(bags):
        return bags[-1]

    @staticmethod
    def __complete_last_bag(bags, max_bag_size, create_empty_sample_func):
        bag = MultiInstanceBagsCollection.__last_bag(bags)
        while len(bag) < max_bag_size:
            bag.add_sample(create_empty_sample_func())

    @staticmethod
    def __is_empty_last_bag(bags):
        return len(MultiInstanceBagsCollection.__last_bag(bags)) == 0

    # endregion

    @classmethod
    def _fill_bags_list_with_linked_text_opinions(cls, bags,
                                                  parsed_rows,
                                                  bag_size,
                                                  create_sample_func,
                                                  create_empty_sample_func):

        assert(isinstance(parsed_rows, list))
        assert(isinstance(bags, list))
        assert(callable(create_sample_func))

        bags.append(Bag(parsed_rows[0].UintLabel))

        for o_ind, parsed_row in enumerate(parsed_rows):
            assert(isinstance(parsed_row, ParsedSampleRow))

            if len(MultiInstanceBagsCollection.__last_bag(bags)) == bag_size:
                # TODO. Use uint_label
                bags.append(Bag(uint_label=parsed_row.UintLabel))

            s = create_sample_func(parsed_row)

            # NOTE: Now we consider that the next appered context always continues the prior.
            prior_opinion = None

            if prior_opinion is not None and not MultiInstanceBagsCollection.__is_empty_last_bag(bags):

                # NOTE: Now we consider that the next appered context always continues the prior.
                is_continued = True

                if not is_continued:
                    MultiInstanceBagsCollection.__complete_last_bag(
                        bags=bags,
                        max_bag_size=bag_size,
                        create_empty_sample_func=create_empty_sample_func)

                    bags.append(Bag(uint_label=parsed_row.UintLabel))

            assert(isinstance(s, InputSampleBase))
            MultiInstanceBagsCollection.__last_bag(bags).add_sample(s)

        if MultiInstanceBagsCollection.__is_empty_last_bag(bags):
            del bags[-1]
            return

        MultiInstanceBagsCollection.__complete_last_bag(
            bags=bags,
            max_bag_size=bag_size,
            create_empty_sample_func=create_empty_sample_func)
