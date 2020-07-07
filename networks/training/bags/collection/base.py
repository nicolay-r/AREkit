class BagsCollection(object):

    def __init__(self, bags):
        assert(isinstance(bags, list))
        self._bags = bags

    @classmethod
    def from_formatted_samples(cls,
                               samples,
                               bag_size,
                               create_sample_func,
                               create_empty_sample_func,
                               text_opinion_helper,
                               shuffle):
        raise NotImplementedError()

    # TODO. To be removed
    # TODO. To be removed
    # TODO. To be removed
    @classmethod
    def from_linked_text_opinions(cls,
                                  text_opinion_collection,
                                  bag_size,
                                  create_sample_func,
                                  create_empty_sample_func,
                                  text_opinion_helper,
                                  shuffle):
        raise NotImplementedError()

    def __iter__(self):
        for bag in self._bags:
            yield bag

    def __len__(self):
        return len(self._bags)
