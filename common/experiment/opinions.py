import collections

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.labels.base import NeutralLabel
from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.linked.text_opinions.collection import LinkedTextOpinionCollection
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.model.sample import InputSampleBase
from arekit.common.news.base import News
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.text_opinion import TextOpinion


# region private methods


def __iter_opinion_collections(opin_operations, doc_id, data_type):
    assert(isinstance(opin_operations, OpinionOperations))
    assert(isinstance(doc_id, int))
    assert(isinstance(data_type, DataType))

    neutral = opin_operations.read_neutral_opinion_collection(doc_id=doc_id,
                                                              data_type=data_type)

    if neutral is not None:
        yield neutral

    if data_type == DataType.Train:
        yield opin_operations.read_etalon_opinion_collection(doc_id=doc_id)


def __check_text_opinion(text_opinion, text_opinion_helper, terms_per_context):
    assert(isinstance(text_opinion, TextOpinion))
    assert(isinstance(text_opinion_helper, TextOpinionHelper))

    return InputSampleBase.check_ability_to_create_sample(
        window_size=terms_per_context,
        text_opinion_helper=text_opinion_helper,
        text_opinion=text_opinion)


def __iter_linked_wraps(experiment, data_type, iter_doc_ids):

    for doc_id in iter_doc_ids:

        news = experiment.DocumentOperations.read_news(doc_id=doc_id)
        assert(isinstance(news, News))

        opinions_it = __iter_opinion_collections(opin_operations=experiment.OpinionOperations,
                                                 doc_id=doc_id,
                                                 data_type=data_type)

        for opinions in opinions_it:
            for linked_wrap in news.iter_wrapped_linked_text_opinions(opinions=opinions):
                yield linked_wrap


# endregions


def extract_text_opinions(experiment,
                          data_type,
                          iter_doc_ids,
                          text_opinion_helper,
                          terms_per_context):
    """
    Extracting text-level opinions based on doc-level opinions in documents,
    obtained by information in experiment.

    NOTE:
    1. Assumes to provide the same label (doc level opinion) onto related text-level opinions.
    """
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, DataType))
    assert(isinstance(terms_per_context, int))
    assert(isinstance(iter_doc_ids, collections.Iterable))
    assert(isinstance(text_opinion_helper, TextOpinionHelper))
    assert(terms_per_context > 0)

    linked_text_opinions = LinkedTextOpinionCollection()

    for linked_wrap in __iter_linked_wraps(experiment, data_type=data_type, iter_doc_ids=iter_doc_ids):
        linked_text_opinions.try_add_linked_text_opinions(
            linked_text_opinions=linked_wrap,
            check_opinion_correctness=lambda text_opinion: __check_text_opinion(
                text_opinion=text_opinion,
                text_opinion_helper=text_opinion_helper,
                terms_per_context=terms_per_context))

    return linked_text_opinions


def compose_opinion_collection(create_collection_func,
                               linked_data_iter,
                               labels_helper,
                               to_opinion_func,
                               label_calc_mode):
    """
    to_opinion_func: (item, label) -> opinion
    """
    assert(callable(create_collection_func))
    assert(isinstance(linked_data_iter, collections.Iterable))
    assert(isinstance(labels_helper, LabelsHelper))
    assert(callable(to_opinion_func))

    collection = create_collection_func()
    assert(isinstance(collection, OpinionCollection))

    for linked in linked_data_iter:
        assert(isinstance(linked, LinkedDataWrapper))

        agg_label = labels_helper.aggregate_labels(
            labels_list=list(linked.iter_labels()),
            label_creation_mode=label_calc_mode)

        agg_opinion = to_opinion_func(linked.First, agg_label)

        assert(isinstance(agg_opinion, Opinion))

        if isinstance(agg_opinion.Sentiment, NeutralLabel):
            continue

        if collection.has_synonymous_opinion(agg_opinion):
            continue

        collection.add_opinion(agg_opinion)

    return collection
