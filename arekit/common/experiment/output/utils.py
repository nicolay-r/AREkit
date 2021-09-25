import collections

from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


def fill_opinion_collection(collection, linked_data_iter, labels_helper, to_opinion_func,
                            label_calc_mode, supported_labels=None):
    """ to_opinion_func: (item, label) -> opinion
    """
    assert(isinstance(collection, OpinionCollection))
    assert(isinstance(linked_data_iter, collections.Iterable))
    assert(isinstance(labels_helper, LabelsHelper))
    assert(callable(to_opinion_func))
    assert(isinstance(supported_labels, set) or supported_labels is None)

    for linked in linked_data_iter:
        assert(isinstance(linked, LinkedDataWrapper))

        agg_label = labels_helper.aggregate_labels(
            labels_list=list(linked.iter_labels()),
            label_calc_mode=label_calc_mode)

        agg_opinion = to_opinion_func(linked.First, agg_label)

        assert(isinstance(agg_opinion, Opinion))

        if supported_labels is not None:
            if agg_opinion.Sentiment not in supported_labels:
                continue

        if collection.has_synonymous_opinion(agg_opinion):
            continue

        collection.add_opinion(agg_opinion)

    return collection
