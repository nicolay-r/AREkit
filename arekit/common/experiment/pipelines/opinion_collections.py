from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.linkage.base import LinkedDataWrapper
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_iter import FilterPipelineItem
from arekit.common.pipeline.item_map import MapPipelineItem


# region private functions


def __create_labeled_opinion(item, label):
    assert(isinstance(item, Opinion))
    return Opinion(source_value=item.SourceValue,
                   target_value=item.TargetValue,
                   sentiment=label)


def __linkages_to_opinions(linkages_iter, labels_helper, label_calc_mode):

    for linkage in linkages_iter:
        assert (isinstance(linkage, LinkedDataWrapper))

        agg_label = labels_helper.aggregate_labels(
            labels_list=list(linkage.iter_labels()),
            label_calc_mode=label_calc_mode)

        yield __create_labeled_opinion(linkage.First, agg_label)


def __create_and_fill_opinion_collection(opinions_iter, collection, supported_labels):
    assert(isinstance(collection, OpinionCollection))
    assert(isinstance(supported_labels, set) or supported_labels is None)

    for opinion in opinions_iter:
        assert(isinstance(opinion, Opinion))

        if supported_labels is not None:
            if opinion.Sentiment not in supported_labels:
                continue

        if collection.has_synonymous_opinion(opinion):
            continue

        collection.add_opinion(opinion)

        yield collection

# endregion


def output_to_opinion_collections(opin_ops, doc_ids_set, labels_scaler,
                                  iter_opinion_linkages_func,
                                  label_calc_mode, supported_labels):
    """ Opinion collection generation pipeline.
    """
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(labels_scaler, BaseLabelScaler))
    assert(isinstance(label_calc_mode, LabelCalculationMode))
    assert(isinstance(supported_labels, set) or supported_labels is None)
    assert(callable(iter_opinion_linkages_func))

    # Opinion collections iterator pipeline
    return BasePipeline([
        FilterPipelineItem(filter_func=lambda doc_id: doc_id in doc_ids_set),

        # Iterate opinion linkages.
        MapPipelineItem(lambda doc_id: (doc_id, iter_opinion_linkages_func(doc_id))),

        # Convert linkages to opinions.
        MapPipelineItem(lambda doc_id, linkages_iter:
                        (doc_id, __linkages_to_opinions(linkages_iter=linkages_iter,
                                                        labels_helper=SingleLabelsHelper(labels_scaler),
                                                        label_calc_mode=label_calc_mode))),

        # Filling opinion collection.
        MapPipelineItem(lambda doc_id, opinions_iter:
                        (doc_id,
                         __create_and_fill_opinion_collection(
                             opinions_iter=opinions_iter,
                             collection=opin_ops.create_opinion_collection(),
                             supported_labels=supported_labels))),
    ])
