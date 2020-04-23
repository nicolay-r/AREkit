import numpy as np
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection, TextOpinion
from arekit.common.model.sample import InputSampleBase
from arekit.networks.context.training.bags.bag import Bag
from arekit.networks.context.training.bags.collection import BagsCollection


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
            shuffle):
        assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))
        assert(isinstance(data_type, unicode))
        assert(isinstance(max_bag_size, int) and max_bag_size > 0)
        assert(callable(create_sample_func))
        assert(callable(create_empty_sample_func))
        assert(isinstance(shuffle, bool))

        def last_bag():
            return bags[-1]

        def complete_last_bag():
            bag = last_bag()
            while len(bag) < max_bag_size:
                bag.add_sample(create_empty_sample_func())

        def is_empty_last_bag():
            return len(last_bag()) == 0

        def is_context_continued(c_rel, p_rel):
            end_type = EntityEndType.Source
            return TextOpinionHelper.extract_entity_sentence_index(p_rel, end_type=end_type) + 1 == \
                   TextOpinionHelper.extract_entity_sentence_index(c_rel, end_type=end_type)

        bags = []

        for text_opinions in text_opinion_collection.iter_by_linked_text_opinions():

            bags.append(Bag(label=text_opinions[0].Sentiment))
            for o_ind, opinion in enumerate(text_opinions):
                assert(isinstance(opinion, TextOpinion))

                if len(last_bag()) == max_bag_size:
                    bags.append(Bag(label=opinion.Sentiment))

                s = create_sample_func(opinion)

                prior_opinion = text_opinions[o_ind - 1] if o_ind > 0 else None
                if prior_opinion is not None and not is_empty_last_bag():
                    if not is_context_continued(c_rel=opinion, p_rel=prior_opinion):
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

