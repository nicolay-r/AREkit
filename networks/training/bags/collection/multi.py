import numpy as np

from arekit.common.linked.base import is_context_continued
from arekit.common.linked.text_opinions.collection import LinkedTextOpinionCollection
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.model.sample import InputSampleBase
from arekit.common.text_opinions.text_opinion import TextOpinion

from arekit.networks.training.bags.bag import Bag
from arekit.networks.training.bags.collection.base import BagsCollection


class MultiInstanceBagsCollection(BagsCollection):
    """
    Has a different algo of bags completion.
    May contain a various amount of instances (samples) within a bag.
    """

    @classmethod
    def from_linked_text_opinions(
            cls,
            text_opinion_collection,
            data_type,
            max_bag_size,
            create_sample_func,
            create_empty_sample_func,
            text_opinion_helper,
            shuffle):
        assert(isinstance(text_opinion_collection, LinkedTextOpinionCollection))
        assert(isinstance(data_type, unicode))
        assert(isinstance(max_bag_size, int) and max_bag_size > 0)
        assert(callable(create_sample_func))
        assert(callable(create_empty_sample_func))
        assert(isinstance(text_opinion_helper, TextOpinionHelper))
        assert(isinstance(shuffle, bool))

        def last_bag():
            return bags[-1]

        def complete_last_bag():
            bag = last_bag()
            while len(bag) < max_bag_size:
                bag.add_sample(create_empty_sample_func())

        def is_empty_last_bag():
            return len(last_bag()) == 0

        bags = []

        for linked_wrap in text_opinion_collection.iter_wrapped_linked_text_opinions():

            bags.append(Bag(label=linked_wrap.First.Sentiment))

            for o_ind, opinion in enumerate(linked_wrap):
                assert(isinstance(opinion, TextOpinion))

                if len(last_bag()) == max_bag_size:
                    bags.append(Bag(label=opinion.Sentiment))

                s = create_sample_func(opinion)

                prior_opinion = linked_wrap.get_prior_opinion_by_index(index=o_ind)
                if prior_opinion is not None and not is_empty_last_bag():
                    is_continued = is_context_continued(text_opinion_helper=text_opinion_helper,
                                                        cur_opinion=opinion,
                                                        prev_opinion=prior_opinion)
                    if not is_continued:
                        complete_last_bag()
                        bags.append(Bag(label=opinion.Sentiment))

                assert(isinstance(s, InputSampleBase))
                last_bag().add_sample(s)

            if is_empty_last_bag():
                bags = bags[:-1]
                continue

            complete_last_bag()

        if shuffle:
            np.random.shuffle(bags)

        return cls(bags)

