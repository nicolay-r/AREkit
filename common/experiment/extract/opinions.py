import collections

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.labels.base import NeutralLabel
from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.model.sample import InputSampleBase
from arekit.common.news.base import News
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


# region private methods

def __iter_linked_text_opinion_lists(news, opin_ops, data_type, filter_text_opinion_func):
    assert(isinstance(news, News))
    assert(callable(filter_text_opinion_func))

    for opinion in opin_ops.iter_opinions_for_extraction(doc_id=news.ID, data_type=data_type):
        linked_text_opinions = news.extract_linked_text_opinions(opinion)
        assert(linked_text_opinions, LinkedTextOpinionsWrapper)

        filtered_text_opinions = filter(filter_text_opinion_func, linked_text_opinions)

        if len(filtered_text_opinions) == 0:
            continue

        yield filtered_text_opinions

# endregions


def iter_linked_text_opins(doc_ops, opin_ops, data_type, parsed_news_it, terms_per_context):
    """
    Extracting text-level opinions based on doc-level opinions in documents,
    obtained by information in experiment.

    NOTE:
    1. Assumes to provide the same label (doc level opinion) onto related text-level opinions.
    """
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(data_type, DataType))
    assert(isinstance(parsed_news_it, collections.Iterable))

    curr_id = 0

    for parsed_news in parsed_news_it:

        linked_text_opinion_lists = __iter_linked_text_opinion_lists(
            news=doc_ops.read_news(doc_id=parsed_news.RelatedNewsID),
            data_type=data_type,
            opin_ops=opin_ops,
            filter_text_opinion_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                parsed_news=parsed_news,
                text_opinion=text_opinion,
                window_size=terms_per_context))

        for linked_text_opinion_list in linked_text_opinion_lists:

            # Assign IDs.
            for text_opinion in linked_text_opinion_list:
                text_opinion.set_text_opinion_id(curr_id)
                curr_id += 1

            yield parsed_news, LinkedTextOpinionsWrapper(linked_text_opinion_list)


def fill_opinion_collection(collection, linked_data_iter, labels_helper, to_opinion_func, label_calc_mode):
    """ to_opinion_func: (item, label) -> opinion
    """
    assert(isinstance(collection, OpinionCollection))
    assert(isinstance(linked_data_iter, collections.Iterable))
    assert(isinstance(labels_helper, LabelsHelper))
    assert(callable(to_opinion_func))

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
