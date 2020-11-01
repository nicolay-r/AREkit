import collections

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
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


# region private methods


def __iter_opinion_collections(opin_operations, doc_id, data_type):
    assert(isinstance(opin_operations, OpinionOperations))
    assert(isinstance(doc_id, int))
    assert(isinstance(data_type, DataType))

    neutral = opin_operations.try_read_neutral_opinion_collection(doc_id=doc_id,
                                                                  data_type=data_type)

    if data_type == DataType.Train:
        # Providing neutral and sentiment.
        if neutral is not None:
            yield neutral
        yield opin_operations.read_etalon_opinion_collection(doc_id=doc_id)

    elif data_type == DataType.Test:
        # Providing neutrally labeled only
        yield neutral

    else:
        raise NotImplementedError("data_type '{}' does not supported!".format(data_type))


def __iter_linked_wraps(doc_ops, opin_ops, data_type, iter_doc_ids):

    for doc_id in iter_doc_ids:

        news = doc_ops.read_news(doc_id=doc_id)
        assert(isinstance(news, News))

        opinions_it = __iter_opinion_collections(opin_operations=opin_ops,
                                                 doc_id=doc_id,
                                                 data_type=data_type)

        for opinions in opinions_it:
            for linked_wrap in news.iter_wrapped_linked_text_opinions(opinions=opinions):
                yield linked_wrap


# endregions


def extract_text_opinions(doc_ops,
                          opin_ops,
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
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(data_type, DataType))
    assert(isinstance(terms_per_context, int))
    assert(isinstance(iter_doc_ids, collections.Iterable))
    assert(isinstance(text_opinion_helper, TextOpinionHelper))
    assert(terms_per_context > 0)

    linked_text_opinions = LinkedTextOpinionCollection()

    wraps_iter = __iter_linked_wraps(doc_ops=doc_ops,
                                     opin_ops=opin_ops,
                                     data_type=data_type,
                                     iter_doc_ids=iter_doc_ids)

    for linked_wrap in wraps_iter:
        linked_text_opinions.try_add_linked_text_opinions(
            linked_text_opinions=linked_wrap,
            check_opinion_correctness=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                text_opinion=text_opinion,
                text_opinion_helper=text_opinion_helper,
                window_size=terms_per_context))

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