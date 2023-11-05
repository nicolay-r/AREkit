from collections.abc import Iterable

from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.linkage.base import LinkedDataWrapper
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.pipeline.items.iter import FilterPipelineItem
from arekit.common.pipeline.items.map import MapPipelineItem


def __create_labeled_opinion(item, label):
    assert(isinstance(item, Opinion))
    return Opinion(source_value=item.SourceValue,
                   target_value=item.TargetValue,
                   label=label)


def __linkages_to_opinions(linkages_iter, labels_helper, label_calc_mode):
    assert(isinstance(linkages_iter, Iterable))

    for linkage in linkages_iter:
        assert(isinstance(linkage, LinkedDataWrapper))

        agg_label = labels_helper.aggregate_labels(
            labels_list=list(linkage.iter_labels()),
            label_calc_mode=label_calc_mode)

        yield __create_labeled_opinion(linkage.First, agg_label)


def __fill_opinion_collection(opinions_iter, collection, supported_labels):
    assert(isinstance(opinions_iter, Iterable))
    assert(isinstance(collection, OpinionCollection))
    assert(isinstance(supported_labels, set) or supported_labels is None)

    for opinion in opinions_iter:
        assert(isinstance(opinion, Opinion))

        if supported_labels is not None:
            if opinion.Label not in supported_labels:
                continue

        if collection.has_synonymous_opinion(opinion):
            continue

        collection.add_opinion(opinion)

    return collection

# endregion


def text_opinion_linkages_to_opinion_collections_pipeline_part(
        doc_ids_set, labels_scaler, iter_opinion_linkages_func,
        create_opinion_collection_func, label_calc_mode):
    """ Opinion collection generation pipeline.
    """
    assert(isinstance(labels_scaler, BaseLabelScaler))
    assert(isinstance(label_calc_mode, LabelCalculationMode))
    assert(callable(iter_opinion_linkages_func))
    assert(callable(create_opinion_collection_func))

    return [
        # Filter doc-ids.
        FilterPipelineItem(filter_func=lambda doc_id: doc_id in doc_ids_set),

        # Iterate opinion linkages.
        MapPipelineItem(lambda doc_id: (doc_id, iter_opinion_linkages_func(doc_id))),

        # Convert linkages to opinions.
        MapPipelineItem(lambda data:
                        (data[0], __linkages_to_opinions(linkages_iter=data[1],
                                                         labels_helper=SingleLabelsHelper(labels_scaler),
                                                         label_calc_mode=label_calc_mode))),

        # Filling opinion collection.
        MapPipelineItem(lambda data:
                        (data[0],
                         __fill_opinion_collection(
                             opinions_iter=data[1],
                             collection=create_opinion_collection_func(),
                             supported_labels=None))),
    ]
